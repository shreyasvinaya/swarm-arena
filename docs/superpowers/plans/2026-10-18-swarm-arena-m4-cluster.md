# swarm-arena M4 Cluster Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run tool jobs on a Ray cluster with `RayExecutor`, and log and cap LLM calls made inside tools through an optional LiteLLM Proxy gateway, with post-run reconcile.

**Architecture:** `swarm_arena.ray` (extra `ray`) holds `RayExecutor`, a generic remote task that runs the tool as a subprocess in the worker's process group, a Ray state lookup by task name, object-store file transport, and `submit`. The executor commits a dispatch-intent event before it submits, and it cancels in two steps in a shielded scope. The core `_gateway/` package (stdlib `urllib`) checks the strict gateway profile at preflight, gives one virtual key to each tool run that may call an LLM, keeps that run's full reservation charged, reads spend back with `record_call(..., settle=False)`, and reconciles after the run under the run lease. Cooperating tools report calls in `swarm/llm_calls.jsonl`.

**Tech Stack:** Python 3.11+, anyio (asyncio backend), pydantic v2, `ray[default]>=2.59,<3` (extra), stdlib `urllib` and `http.server` (gateway client and test fake), pytest 9, uv, ruff, ty, Sphinx.

**Spec:** `docs/superpowers/specs/2026-10-16-swarm-arena-design.md` (APPROVED r8). Binding roadmap: `docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md` (module map R3, interface registry R4). Read both. This plan owns spec §4.8 `RayExecutor`, §6.1 (Ray schedule clock), §6.5 (Ray runs on resume), §7.3 item 2 (a, b, c), §8.1 gateway tool runs, §9 rule 4 completion, §11 Secrets and Ray mechanics.

## Global Constraints

- Branch `m4-cluster` from `main` with M1a to M3 merged. One commit per task. Commit messages: imperative subject, body with what and why, **no AI or Claude co-author line, no "generated with" text**.
- Python `>=3.11`, POSIX only. anyio on the **asyncio backend only**.
- Extra: `ray = ["ray[default]>=2.59,<3"]`. The gateway client uses **only the stdlib** (`urllib`). No other new dependency.
- The core never imports `ray` (ruff `TID251`); only `src/swarm_arena/ray/**` and the Ray test files may. `_cli` imports `swarm_arena.ray` only inside a command handler (R4.13.18).
- Ray: `max_retries=0`, `max_calls=1`, no `setsid` (no `start_new_session`, no `process_group`), no `runtime_env.py_executable`, no Ray Client (`ray://`) for long runs, `ray.init(address="auto")` through `anyio.to_thread.run_sync`, poll `p.wait(timeout=0.2)`, object-store file cap 256 MiB.
- Cancel: `ray.cancel(ref)`, wait `grace_s` + `cancel_margin_s` (default 10.0), `ray.cancel(ref, force=True)`, all in a shielded scope.
- Secrets are never written by the core (journal, logs, artifacts, manifest, arena objects, LLM payloads) and **never put in Ray `runtime_env`**. Secret values are read on the worker from its environment or `RunConfig.secrets.file`.
- Gateway key: `key_alias = "cand-<candidate id>"`, `max_budget` = the tool run's reservation (USD), `duration` = `GatewayConfig.key_duration_s` (86400), `metadata = {candidate_id, run_id, goal_id, team, agent, tags, spend_logs_metadata: {candidate_id, …}}`, `default_estimated_output_tokens` = `GatewayConfig.default_estimated_output_tokens` (4096).
- Gateway environment for a keyed run: `OPENAI_BASE_URL = <url>/v1`, `ANTHROPIC_BASE_URL = <url>`, `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` = the virtual key. Base URLs go in `Job.env`, the key in `Job.secret_values` (R4.7).
- A keyed tool run keeps its **full reservation** charged for the whole run. `terminal_accounting = False` for LiteLLM. Report labels: `observed (completeness unverified)`, `estimated` (and `observed` only for a terminal-accounting adapter).
- Clocks: `settle_timeout_s` 60, `llm_gateway.request_timeout_s` 600, `schedule_timeout_s` 600, `lease_ttl_s` 60.
- **Must NOT:** add public names outside `swarm_arena.ray`, except two fields: `ToolOutput.llm_calls_file` (Task 3) and `Job.drop_env` (Task 13). Each task that adds one also amends the roadmap's M4 **Must NOT** and R5.4 item 4 in its commit, with the reason. `RunConfig.schedule_timeout_s` reaches executors through a private context variable (Task 7), so `Job`, `CheckContext`, and `JudgeContext` do not change. Also do not change the LLM layer beyond the shared recording path (`_journal/llm_record.py`), and do not add container or SSH executors.
- Facades (R0, M1a Task 1): a facade holds only a docstring, imports, `__all__`, and one `_publish(__name__, __all__)` call (M1a `_facade.publish`). `swarm_arena/ray/__init__.py` gets that call in Task 7. No facade sets `__module__` by hand.
- Owner rules: uv (`uv add`, `uv run`), ruff lint and format, ty, every function and method fully annotated (tests too), pytest with tests mirroring `src/` (R3), TDD, numpydoc docstrings with an `Examples` section on every public object, README and docs updated in the task that adds the feature.
- Test markers: tests that need a Ray cluster carry `pytestmark = pytest.mark.ray`. `gateway_live` tests are manual. Default `uv run pytest` runs neither. A command-line `-m` replaces the `addopts` expression, so the default Ray command is `uv run pytest -m "ray and not s3"`. The one `ray` and `s3` test runs with `-m "ray and s3"` (Task 7).
- A `ray`-marked test in a module that the core CI job collects (`tests/_engine/test_resume.py`, `tests/testing/test_conformance.py`) calls `pytest.importorskip("swarm_arena.ray")` and imports `swarm_arena.ray` inside the test body, because `-m` deselects only after import. Every test that needs Ray and can run in the `core` job skips this way. Never `pytest.importorskip("ray")`: under `pythonpath = ["tests"]` without Ray it imports the `tests/ray/` namespace package and does not skip.
- Docs pages are the M1a stubs: `docs/running-on-ray.md`, `docs/writing-a-connector.md`, `docs/own-files.md`, and `docs/limits-and-safety.md`. The API page `docs/api/ray.md` is new (Task 7), and `docs/api/index.md` lists it in its toctree.

## Review Focus

1. **The gateway virtual key leaks to disk.** A keyed run writes the key value into a journal segment, the local journal file, tool logs, artifacts, `manifest.json`, or an error message. Expected: the value is in none of them. Test: Task 15 `test_virtual_key_never_written`.
2. **A cancel arrives while the Ray task is still pending** (queued, or no node fits). Expected: the tool never starts, `submit` ends within `grace_s + cancel_margin_s`, and no task runs later. Test: Task 7 `test_cancel_while_pending_never_runs_tool`.
3. **The gateway fails in the middle of a run** (`/key/generate`, `/key/block`, or `/spend/logs/v2` returns 500 or times out). Expected: a failed key create means no dispatch, an `error` candidate, and the reservation stays charged. A failed block or read keeps the candidate's result and adds a warning. The swarm goes on. Test: Task 15 `test_gateway_failures_never_stop_the_run`.
4. **`swarm/llm_calls.jsonl` is malformed or hostile** (bad JSON, a JSON array line, negative tokens, NaN cost, more than 10,000 rows, an oversize file). Expected: bad lines are skipped with a candidate warning, and the run never crashes. Test: Task 3 `test_hostile_llm_calls_file_never_crashes`.
5. **The driver dies after it commits the dispatch intent but before `.remote()`**, so no task exists. Expected: resume does not hang, the run is `lost`, and `confirmed_dead` is false. Test: Task 8 `test_resume_dispatch_never_submitted_is_lost`.

---

## File structure

| Path | Responsibility | Task |
|---|---|---|
| `tasks/research/api-litellm-gateway.md` | Checked LiteLLM Proxy and Ray facts | 1 |
| `src/swarm_arena/_exec/job_steps.py` (new row in R3.1) | Job steps shared by both executors: prepare (materialize, `command()`), collect (`collect()` or `result.json`), upload (output files, logs, `llm_calls.jsonl`); `live_request` restores edit contents after a JSON transport (the Ray worker) | 2, 3 |
| `src/swarm_arena/_tooling/llm_calls.py` | `LLMCallRow`, `parse_llm_calls`, `llm_calls_source`, `record_reported`, `PROVIDER_KEYS` | 3 |
| `src/swarm_arena/ray/__init__.py` | Facade; clear `ImportError` without the extra | 4, 7 |
| `src/swarm_arena/ray/_files.py` | `MemoryStore`, input inlining with the cap, result framing | 4 |
| `src/swarm_arena/ray/_task.py` | `TaskSettings`, `job_payload`, `driver_import_roots`, `run_job`, `run_in_worker` | 5 |
| `src/swarm_arena/ray/_state.py` | `TaskInfo`, `find_task`, `worker_dead` | 6 |
| `src/swarm_arena/ray/_executor.py` | `RayExecutor`, `task_options`, `request_shape`, `fits_any_node` | 7 |
| `src/swarm_arena/ray/_submit.py` | `submit_driver`, `driver_entrypoint`, `submit_runtime_env` | 9 |
| `src/swarm_arena/_cli/cluster_cmds.py` | `submit`, `reconcile` commands | 9, 16 |
| `src/swarm_arena/_gateway/adapter.py` | `GatewayAdapter`, `VirtualKey`, `SpendRow`, `ProfileReport`, `GatewayError` | 11 |
| `src/swarm_arena/_gateway/profile.py` | `check_profile` (pure) and the setting names (`None` when absent in 1.105.x) | 11 |
| `src/swarm_arena/_gateway/litellm_proxy.py` | `LiteLLMProxy`, `adapter_for` | 12 |
| `src/swarm_arena/_gateway/keys.py` | Key per tool run, gateway `Job` values, `KeyManager` | 13 |
| `src/swarm_arena/_gateway/reconcile.py` | `reconcile_keys`, `reconcile_run`, `KeyReport`, `ReconcileReport` | 16 |
| `tests/_fakes/gateway.py` | `strict_settings`, `strict_models`, `ABSENT_FAILURES`, `FakeGateway` (stdlib HTTP) | 11, 12 |
| `tests/ray/conftest.py` (new row in R3.2) | `ray_driver` fixture (in-process `ray.init` to the session cluster) | 5 |
| `tests/e2e/test_ray_cancel.py`, `test_ray_dispatch_crash.py`, `test_rosenbrock_ray.py`, `test_gateway_run.py` | M4 exit tests | 7, 8, 10, 17 |

Files that M4 extends (each task adds the `+M4` tag to that file's R3.1 row in the same commit): `_model/tooling.py` (3), `_journal/llm_record.py` (3), `_agent/tool_run.py` (3, 8, 15), `_planning/validate.py` (3, 15), `_exec/local.py` (2, 16), `_exec/supervisor.py` (5), `_exec/executor.py` (7, 13), `_exec/env.py` (13), `_journal/replay.py` (8), `_engine/resume.py` (7, 8, 15), `_cli/main.py` (9), `_engine/preflight.py` (14), `_engine/manifest.py` (14), `_board/trace.py` (14), `_engine/run.py` (7, 14, 15, 16), `_engine/goal.py` (15), `_agent/runtime.py` (15), `_budget/ledger.py` (15).

Roadmap edits, each made in its task's commit, with the reason in the commit body (R0):
- R4.2 `ToolOutput`: `llm_calls_file: FileRef | None = None` (Task 3). Add it to R5.4 item 4, and add it as an exception to the M4 **Must NOT**. Its comment adds: a valid `FileRef` with `path = "llm_calls.jsonl"` (`check_relpath` refuses the reserved `swarm/`) and the file's real `sha256`, also when the file is over 16 MiB and not uploaded. Reason: the reported-calls file is made where the tool runs (possibly a Ray worker), so it must travel to the driver with the output.
- R4.7 `Job`: `drop_env: tuple[str, ...] = ()` (Task 13). Add it to R5.4 item 4, and add it as an exception to the M4 **Must NOT**. Reason: `build_env` runs on the executor side, so the Job must carry the provider key names that a keyed run never gets from the node (spec §7.3 item 2a step 2).
- R4.13.10 `AgentRuntime`: `gateway: KeyManager | None = None`, last field (Task 15).
- R4.13.16: `job_payload(job: Job, settings: TaskSettings, *, blobs: Mapping[str, bytes] | None = None) -> bytes` (Task 5).
- R4.13.16: `submit_driver(address: str | None, entrypoint: str, *, runtime_env: Mapping[str, Any] | None) -> str` (Task 9). Reason: `None` lets Ray read `RAY_ADDRESS`, as `swarm-arena submit` does without `--address`.
- R4.13.18: the CLI addressing rule (Task 16). Run-state commands (`status`, `stop`, `resume`, `reconcile`) take `RUN_DIR`; arena commands take a run id or a namespace plus `--config`. Reason: M1b's `resume` takes `RUN_DIR` and M2's arena commands take an id, so `reconcile` needs one written rule.
- M4 exit criteria: `uv run pytest -m ray -v` becomes `uv run pytest -m "ray and not s3" -v` (Task 7). Reason: a command-line `-m` replaces the `addopts` expression, and the one `ray` and `s3` test needs an S3 endpoint that the Ray workers can reach.

---

### Task 1: Verify the LiteLLM Proxy and Ray facts

**Files:**
- Create: `tasks/research/api-litellm-gateway.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the exact names that Tasks 5 to 7 and 11 to 16 use. Tasks 11 and 12 copy the gateway names into `_gateway/profile.py` and `_gateway/litellm_proxy.py`. Tasks 6 and 7 copy the Ray state names into `ray/_state.py` and `ray/_executor.py`.

- [ ] **Step 1: Get the pinned sources (read only, nothing is executed from them)**

```bash
mkdir -p /tmp/m4-facts
uv run --no-project --with 'litellm[proxy]>=1.105.1,<1.106' python -I -c "import litellm, os; print(litellm.__version__, os.path.dirname(litellm.__file__))"
uv run --no-project --with 'ray[default]>=2.59,<3' python -I -c "import ray, os; print(ray.__version__, os.path.dirname(ray.__file__))"
```

Expected: two lines. Each has the version and the package path in the uv cache. Search those paths with `rg`.

- [ ] **Step 2: Record each fact with its value, default, and `file:line`**

Gateway (LiteLLM Proxy 1.105.x):
1. The setting that turns budget reservations on or off, and its default.
2. The setting that makes budget enforcement fail closed when spend counters are stale or not available.
3. The maximum request duration setting, and where the proxy enforces it. Also record whether the enforced duration bounds a streaming response from end to end (not only the time to the first byte). If it does not, item 3 is `absent`.
4. The spend-log batch write interval (expected `proxy_batch_write_at`), and its default.
5. The setting that requires virtual keys.
6. The pass-through endpoints setting.
7. `/key/generate`: the request fields `key_alias`, `duration` (string format), `max_budget`, `metadata`, and the field that holds `default_estimated_output_tokens`. The response fields for the secret key and for the key id (hashed token).
8. `/key/block` body. `/key/info` query parameter, its `blocked` field, and the status code for a deleted key.
9. `/key/delete` body.
10. `/spend/logs/v2`: the query parameters (key filter, date range, pages). The row fields: the LiteLLM call id, model, prompt and completion tokens, cache tokens, spend, start time, metadata. Do rows stay after the key is deleted?
11. `/model/info`: the price fields (`input_cost_per_token`, `output_cost_per_token`) and the max output token field.
12. The config read endpoint: its path and the response shape (`general_settings`, `litellm_settings`).
13. The health endpoint path.

For items 1 to 6, 22, and 23: when 1.105.x has no such setting, write `absent`, and use the files searched as its Source line.

Ray 2.59:

14. The `state` values of a task in `ray.util.state` (the pending states, running, and the terminal states).
15. `ray.util.state.list_tasks(filters=[("name", "=", name)], detail=True)`: does the name filter work? What are the `worker_id` and `node_id` fields? Does `list_workers` or `get_worker` give `is_alive`?
16. Owner fate sharing: after a SIGKILL of a driver, does the raylet kill the workers of its running tasks, how long does it take, and does the task state become `FAILED`?
17. Does `ray.init(address="auto")` honor `RAY_ADDRESS`? Does a second local driver work with the default token auth of 2.59? Do `ray.util.state.list_tasks` and `JobSubmissionClient(...).submit_job` work from a second local process under that default token auth, and which environment variables do they need?
18. Does `RemoteFunction._default_options` hold `max_retries` and `max_calls`?
19. Does `JobSubmissionClient.submit_job` return the submission id as `str`?
20. `ray.nodes()`: the field that tells if a node is alive (expected `Alive`), and the field with its **total** resources (expected `Resources`, with the keys `CPU`, `GPU`, `memory`, and custom names).
21. Soft cancel with a normal return: on a local `ray start --head`, run a `max_calls=1` task that polls `p.wait(timeout=0.2)`, catches the `KeyboardInterrupt` of `ray.cancel(ref)`, and returns bytes. Record whether `ray.get(ref)` returns those bytes or raises `TaskCancelledError`, with the source lines (`python/ray/_raylet.pyx`, `src/ray/core_worker/task_manager.cc`) and the commands.

Gateway, more (LiteLLM Proxy 1.105.x):

22. How 1.105.x bounds output tokens per key: the key field or proxy setting that caps a request's `max_tokens` (or fills it when the request omits it) for the budget reservation; what the proxy does when a request omits `max_tokens` or asks for more than the bound (reject, clamp, or reserve at the model maximum); how `default_estimated_output_tokens` takes part.
23. The routes that budget reservations cover, and the setting that limits the proxy to those routes, with its default.

For each item, write a `- **Source:** <file>:<line>` line. Use the format of `tasks/research/api-ray.md`. For items 16, 17, and 21, run the check on a local `ray start --head` and record the commands.
**Blocker rule:** if 1.105.x has no `default_estimated_output_tokens` on a key (item 7), stop. Report that blocker in one sentence, because §7.3 item 1 requires that field. If item 21 raises `TaskCancelledError`, stop. Report in one sentence that Ray 2.59 cannot deliver the partial `ToolOutput` that spec §11 requires a soft-cancelled task to return.

- [ ] **Step 3: Check that the file is complete**

Run: `rg -c '^- \*\*Source:' tasks/research/api-litellm-gateway.md && rg -n 'TBD|\?\?\?' tasks/research/api-litellm-gateway.md`
Expected: `23`, and no matches for the second pattern.

- [ ] **Step 4: Commit**

```bash
git add tasks/research/api-litellm-gateway.md
git commit -m "docs: record LiteLLM Proxy and Ray facts for M4" -m "Exact setting and endpoint names for the strict gateway profile and the key, spend, and model APIs, plus Ray state, fate-sharing, and options facts, each with its source line."
```

---

### Task 2: Share the job steps between executors

**Files:**
- Create: `src/swarm_arena/_exec/job_steps.py`
- Modify: `src/swarm_arena/_exec/local.py` (call the new functions; behavior unchanged)
- Modify: `docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md` (R3.1: add the `_exec/job_steps.py` row, test `tests/_exec/test_job_steps.py`, M4; tag `_exec/local.py` `+M4`)
- Test: `tests/_exec/test_job_steps.py`

**Interfaces:**
- Consumes: `Job`, `Job.Stage` (R4.7), `ArtifactStore`, `ArtifactStore.get_bytes`, `NotFound` (R4.13.1, R4.13.2), `materialize`, `collect_output_files` (R4.13.9), `note_in_log`, `command_output_from_result` (M1a), `Tool.command`, `Tool.collect` (R4.2), `Command`, `CompletedRun`, `ToolOutput`, `ToolRequest` with its `edits` serializer, the reserved `sha256:<64 hex>` reference form, and the executor transport rule (R4.2, M1a Task 6; no change and no roadmap edit here), `put_edits`, `edit_hashes` (R4.13.9, tests only).
- Produces (used by `LocalExecutor` and `ray/_task.py`; `live_request` by `ray/_task.py` only):

```python
# _exec/job_steps.py
def prepare_job(job: Job, workspace: Path, artifacts: ArtifactStore) -> tuple[Command, Tool | None]: ...
    # materialize job.stage; kind="tool": tool = job.tool.build(Tool), command = tool.command(job.request, workspace);
    # kind="command": (job.command, None). Command.timeout_s is set to job.timeout_s.
def resolve_argv(command: Command) -> list[str]: ...
    # a leading "python" becomes command.python, else sys.executable (the M1a rule)
def collect_output(job: Job, tool: Tool | None, workspace: Path, run: CompletedRun) -> ToolOutput: ...
    # tool: tool.collect(job.request, workspace, run); command: swarm/result.json read by the M1a read_result_json rule
    # (through read_regular_file); an exception → ToolOutput(status=ERROR) and note_in_log(stderr, traceback) (the M1a rule)
def upload_outputs(output: ToolOutput, workspace: Path, run: CompletedRun, artifacts: ArtifactStore) -> ToolOutput: ...
    # declared output files (a ValueError → status ERROR, files=[], log note); both logs, read through the M1a
    # read_regular_file rule → logs_ref "sha256:<out>:<err>"; started_at and ended_at from run
def live_request(request: ToolRequest, artifacts: ArtifactStore) -> ToolRequest: ...
    # the inverse of the R4.2 edits serializer, for a request that crossed a JSON transport (the Ray payload): each edits
    # value that fully matches ^sha256:[0-9a-f]{64}$ becomes artifacts.get_bytes(<the 64 hex>).decode("utf-8") (strict);
    # any other value is live content and stays (live entry points refuse the reserved form as content, so no value is
    # ambiguous and a second call changes nothing). Returns request.model_copy(update={"edits": <same keys, same order>});
    # other fields unchanged. NotFound, a failed hash check, or a UnicodeDecodeError →
    # ValueError(f"edit {path!r}: stored reference {value} could not be restored: {type(e).__name__}: {e}") from e
```

`LocalExecutor` never calls `live_request`: its `Job.request` is the in-memory request that `execute_tool_run` or `check_tool` built, with live contents that `validate_edits` checked. Only the Ray worker gets a request with references (Task 5 `run_in_worker` step 2).

The split keeps the M1a order: on a timeout or a cancel, `LocalExecutor` calls `collect_output` once, in `before_kill` (before SIGKILL), through `anyio.to_thread.run_sync(..., abandon_on_cancel=True)` under the supervisor's `BEFORE_KILL_TIMEOUT_S` (R4.13.8), and passes that stored output to `upload_outputs`. After a normal exit it calls `collect_output(job, tool, workspace, run)` once. The `read_regular_file` and `read_result_json` rules move from `LocalExecutor` unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_prepare_job_tool_kind_returns_command_and_tool(tmp_path: Path) -> None:
    command, tool = prepare_job(_tool_job(timeout_s=5.0), tmp_path / "ws", _artifacts(tmp_path))
    assert isinstance(tool, Tool) and command.timeout_s == 5.0

def test_prepare_job_command_kind_uses_job_command(tmp_path: Path) -> None:
    job = _command_job(Command(argv=["python", "-c", "pass"]))
    assert prepare_job(job, tmp_path / "ws", _artifacts(tmp_path)) == (job.command, None)

def test_resolve_argv_replaces_leading_python() -> None:
    assert resolve_argv(Command(argv=["python", "x.py"], python="/opt/py"))[0] == "/opt/py"
    assert resolve_argv(Command(argv=["python", "x.py"]))[0] == sys.executable
    assert resolve_argv(Command(argv=["/bin/echo"]))[0] == "/bin/echo"

def test_upload_outputs_sets_logs_ref(tmp_path: Path) -> None:
    out = upload_outputs(ToolOutput(status=RunStatus.OK), ws, _completed(ws, stdout=b"o", stderr=b"e"), artifacts)
    assert out.logs_ref == f"sha256:{sha256_hex(b'o')}:{sha256_hex(b'e')}"

def test_collect_output_command_kind_parses_result_json(tmp_path: Path) -> None:
    (ws / "swarm" / "result.json").write_text('{"data": {"x": 1}}')
    assert collect_output(job, None, ws, _completed(ws)).data == {"x": 1}

EDITS = {"p.txt": "hello", "q.txt": "sha256:hello"}      # ordinary text, and content that only looks like a reference

def test_live_request_restores_transported_edits(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    put_edits(EDITS, artifacts)
    sent = ToolRequest.model_validate_json(ToolRequest(inputs={"n": 1}, edits=EDITS).model_dump_json())
    assert sent.edits == {p: f"sha256:{h}" for p, h in edit_hashes(EDITS).items()}     # what the JSON transport carries
    live = live_request(sent, artifacts)
    assert (live.edits, live.inputs) == (EDITS, {"n": 1})

def test_live_request_keeps_live_content(tmp_path: Path) -> None:
    assert live_request(ToolRequest(edits=EDITS), _artifacts(tmp_path)).edits == EDITS   # empty store: nothing is read

def test_live_request_unrestorable_reference_raises(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    not_utf8 = artifacts.put_bytes(b"\xff\xfe")
    for value in (f"sha256:{sha256_hex(b'absent')}", f"sha256:{not_utf8}"):
        with pytest.raises(ValueError, match=r"^edit 'p.txt': stored reference sha256:[0-9a-f]{64} could not be restored"):
            live_request(ToolRequest(edits={"p.txt": value}), artifacts)
```

`_tool_job`, `_command_job`, `_artifacts`, and `_completed` are small helpers in this test file. `_tool_job` uses a `ScriptTool` ref from `_fakes.tools`.

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/_exec/test_job_steps.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._exec.job_steps'`

- [ ] **Step 3: Move the code into `src/swarm_arena/_exec/job_steps.py`**

Move the existing code from `_exec/local.py`; do not rewrite it. Move the code that materializes the stage, builds the `Command`, replaces a leading `python`, calls `collect()` (or parses `swarm/result.json`), collects the output files, and uploads the logs. `LocalExecutor.submit` then calls `prepare_job` → `ProcessSupervisor.run(resolve_argv(command), ..., before_kill=<stores collect_output(job, tool, workspace, provisional), run in a thread with abandon_on_cancel=True>)` → `upload_outputs(<the stored partial output after a stop; ToolOutput(status=run.status) after a stop with no stored output (collect not tried again, M1a Task 17 step 7); else collect_output(job, tool, workspace, run) after a normal exit>, workspace, run, artifacts)`. The process start, the environment, and the cancel handling stay in `local.py`. Then add `live_request` with the signature above; it is the only new code in this task, and `LocalExecutor` does not call it.

- [ ] **Step 4: Run the tests to check that they pass, with no change in local behavior**

Run: `uv run pytest tests/_exec -v && uv run ruff check src/swarm_arena/_exec && uv run ty check`
Expected: PASS. All of `tests/_exec/test_local.py` and `tests/_exec/test_supervisor.py` stay green.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_exec/job_steps.py src/swarm_arena/_exec/local.py tests/_exec/test_job_steps.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "refactor: share job preparation and collection between executors" -m "Move stage materialization, command building, collect, output files, and log upload from LocalExecutor into _exec/job_steps.py, so RayExecutor's worker task runs the same steps. No behavior change. Add live_request, which turns the sha256 edit references of a request that crossed the Ray JSON transport back into contents, so the worker can give Tool.command and Tool.collect the request that LocalExecutor gives them. Roadmap R3.1 gains the new module row."
```

---
### Task 3: Cooperating tools: `swarm/llm_calls.jsonl` and `llm_calls_source`

**Files:**
- Create: `src/swarm_arena/_tooling/llm_calls.py`
- Modify: `src/swarm_arena/_model/tooling.py` (`ToolOutput.llm_calls_file`)
- Modify: `src/swarm_arena/_exec/job_steps.py` (`upload_outputs` sets `llm_calls_file`)
- Modify: `src/swarm_arena/_agent/tool_run.py` (record the rows, set `llm_calls_source`)
- Modify: `src/swarm_arena/_journal/llm_record.py` (a `None` payload is not stored)
- Modify: `src/swarm_arena/_planning/validate.py` (warning: providers and no gateway)
- Modify: `docs/own-files.md`, `docs/writing-a-connector.md`, roadmap (R4.2, R5.4 item 4, the M4 **Must NOT** exception, `+M4` tags)
- Test: `tests/_tooling/test_llm_calls.py`, `tests/_exec/test_job_steps.py`, `tests/_agent/test_tool_run.py`, `tests/_journal/test_llm_record.py`, `tests/_planning/test_validate.py`

**Interfaces:**
- Consumes: `record_call(..., settle=False)` (R4.13.19), `Usage`, `FileRef`, `StoredModel`, `Candidate.llm_call_ids`, `Candidate.llm_calls_source`, `Candidate.warnings` (R4.6), `Tool.providers()`.
- Produces:

```python
# _model/tooling.py (+M4): ToolOutput gains
    llm_calls_file: FileRef | None = None       # [plan] swarm/llm_calls.jsonl in artifact_root, set by upload_outputs; None if absent;
        # a valid FileRef: path = LLM_CALLS_REF_PATH, sha256 = the file's real hash, size set. Over MAX_LLM_CALLS_BYTES
        # the file is hashed but not uploaded, and the driver never fetches it (it checks size first)

# _tooling/llm_calls.py
LLM_CALLS_PATH: Final = "swarm/llm_calls.jsonl"      # where the tool writes it, relative to the workspace
LLM_CALLS_REF_PATH: Final = "llm_calls.jsonl"        # [plan] FileRef.path of llm_calls_file: check_relpath refuses the
                                                     # reserved top-level swarm/ (M1a Tasks 4 and 6)
MAX_LLM_CALLS_BYTES: Final = 16 * 1024 * 1024        # [plan]
MAX_LLM_CALL_ROWS: Final = 10_000                     # [plan] non-empty lines read, valid or not
MAX_LLM_CALL_WARNINGS: Final = 20                     # [plan] line warnings; more bad lines get one summary warning
PROVIDER_KEYS: Final[frozenset[str]] = frozenset({    # [plan] provider key names; Task 13 replaces two of them
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "DEEPSEEK_API_KEY",
    "FIREWORKS_AI_API_KEY", "MOONSHOT_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
    "GROQ_API_KEY", "XAI_API_KEY", "TOGETHERAI_API_KEY", "AZURE_API_KEY", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY"})
class LLMCallRow(StoredModel):                        # model_config extra="ignore": unknown keys are dropped
    model: str
    provider: str | None = None                       # None → text before the first "/" of model, else "unknown"
    usage: Usage = Usage()                            # every token count >= 0
    cost_usd: float = 0.0                             # finite, >= 0
    latency_ms: float | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    litellm_call_id: str | None = None
    request: JsonValue = None
    response: JsonValue = None
def parse_llm_calls(data: bytes) -> tuple[list[LLMCallRow], list[str]]: ...   # (rows, warnings); never raises
def hash_regular_file(workspace: Path, rel: str) -> tuple[str, int]: ...    # (sha256 hex, size): the read_regular_file
    # open checks (no symlinked component, O_NOFOLLOW | O_NONBLOCK, fstat S_ISREG; same ValueError), size from that fstat,
    # then sha256 over 1 MiB chunks; never holds the whole file in memory
def may_call_llm(providers: frozenset[str], secret_env: Sequence[str]) -> bool: ...
def llm_calls_source(*, keyed: bool, reported_rows: int,
                     may_call: bool) -> Literal["gateway", "reported", "none", "unknown"]: ...
async def record_reported(rows: Sequence[LLMCallRow], *, metadata: Mapping[str, Any]) -> list[LLMCall]: ...
    # record_call(settle=False, role "tool" in metadata, cache_hit=False, tool_call_ids=()); a None return is a duplicate
```

`llm_calls_source` rules [plan]: `keyed` → `"gateway"`; else `reported_rows > 0` → `"reported"`; else `may_call` → `"unknown"`; else `"none"`. The core never sets `"langfuse"` in v1, because it cannot see a tool's own Langfuse traces. The docs say this.

Warning texts (exact): `f"swarm/llm_calls.jsonl line {n}: {reason}; line skipped"`, `f"swarm/llm_calls.jsonl: {k} more bad lines skipped"`, `f"swarm/llm_calls.jsonl has more than 10000 rows; rows after 10000 skipped"`, `f"swarm/llm_calls.jsonl is {size} bytes, over the 16 MiB limit; not collected"`. The `validate_plan` warning: `f"agent {address}: tool declares providers {sorted(providers)} but there is no llm_gateway; its LLM calls are logged only if it writes swarm/llm_calls.jsonl (llm_calls_source = \"unknown\")"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_tooling/test_llm_calls.py
def test_parse_valid_rows() -> None:
    rows, warnings = parse_llm_calls(b'{"model": "openai/gpt-6", "usage": {"input_tokens": 10}, "cost_usd": 0.01}\n'
                                     b'{"schema_version": 1, "model": "local-llm", "litellm_call_id": "abc"}\n')
    assert warnings == [] and [r.schema_version for r in rows] == [1, 1]
    assert rows[0].provider == "openai" and rows[1].provider == "unknown" and rows[1].litellm_call_id == "abc"

def test_parse_skips_bad_lines() -> None:
    rows, warnings = parse_llm_calls(b'{"model": "a/b"}\nnot json\n{"model": "a/b", "cost_usd": NaN}\n')
    assert len(rows) == 1 and any("line 2" in w for w in warnings) and any("line 3" in w for w in warnings)

def test_parse_row_cap() -> None:
    rows, warnings = parse_llm_calls(b'{"model": "a/b"}\n' * 10_001)
    assert len(rows) == 10_000 and warnings == ["swarm/llm_calls.jsonl has more than 10000 rows; rows after 10000 skipped"]

def test_parse_ignores_unknown_keys() -> None:
    assert parse_llm_calls(b'{"model": "a/b", "vendor_field": 1}\n')[0][0].model == "a/b"

def test_hostile_llm_calls_file_never_crashes() -> None:          # Review Focus 4
    data = b"\n".join([b"", b"[1, 2]", b'{"model": "a/b", "usage": {"input_tokens": -5}}', b'{"model": 7}',
                       b'{"model": "a/b", "cost_usd": Infinity}', b"{" * 100_000, b'{"model": "ok/x"}'])
    rows, warnings = parse_llm_calls(data)
    assert [r.model for r in rows] == ["ok/x"] and len(warnings) == 5        # the empty line is skipped silently

def test_parse_bad_line_warnings_capped() -> None:
    rows, warnings = parse_llm_calls(b"x\n" * 1000)
    assert rows == [] and len(warnings) == 21
    assert warnings[-1] == "swarm/llm_calls.jsonl: 980 more bad lines skipped"

@pytest.mark.parametrize(("keyed", "rows", "may_call", "source"), [
    (True, 3, True, "gateway"), (False, 2, True, "reported"), (False, 0, True, "unknown"), (False, 0, False, "none")])
def test_llm_calls_source(keyed: bool, rows: int, may_call: bool, source: str) -> None:
    assert llm_calls_source(keyed=keyed, reported_rows=rows, may_call=may_call) == source

def test_may_call_llm() -> None:
    assert may_call_llm(frozenset({"openai"}), ()) and may_call_llm(frozenset(), ["GEMINI_API_KEY"])
    assert not may_call_llm(frozenset(), ["MY_TOKEN"])

def test_hash_regular_file(tmp_path: Path) -> None:
    (tmp_path / "swarm").mkdir(); (tmp_path / "swarm" / "llm_calls.jsonl").write_bytes(b"x" * (3 * 1024 * 1024 + 1))
    assert hash_regular_file(tmp_path, LLM_CALLS_PATH) == (sha256_hex(b"x" * (3 * 1024 * 1024 + 1)), 3 * 1024 * 1024 + 1)
    (tmp_path / "swarm" / "link.jsonl").symlink_to(tmp_path / "swarm" / "llm_calls.jsonl")
    with pytest.raises(ValueError, match="is not a regular file"):
        hash_regular_file(tmp_path, "swarm/link.jsonl")

# tests/_exec/test_job_steps.py
def test_upload_outputs_sets_llm_calls_file(tmp_path: Path) -> None:
    (ws / "swarm" / "llm_calls.jsonl").write_bytes(b'{"model": "a/b"}\n')
    out = upload_outputs(ToolOutput(status=RunStatus.OK), ws, _completed(ws), artifacts)
    assert out.llm_calls_file == FileRef(name="llm_calls.jsonl", path="llm_calls.jsonl",
                                         sha256=sha256_hex(b'{"model": "a/b"}\n'), size=17)
    assert artifacts.get_bytes(out.llm_calls_file.sha256) == b'{"model": "a/b"}\n'

def test_upload_outputs_skips_oversize_llm_calls_file(tmp_path: Path) -> None:
    (ws / "swarm" / "llm_calls.jsonl").write_bytes(b"x" * (17 * 1024 * 1024))
    out = upload_outputs(ToolOutput(status=RunStatus.OK), ws, _completed(ws, stdout=b"o", stderr=b"e"), artifacts)
    assert out.llm_calls_file.size == 17 * 1024 * 1024 and out.llm_calls_file.path == "llm_calls.jsonl"
    assert out.llm_calls_file.sha256 == sha256_hex(b"x" * (17 * 1024 * 1024))       # real hash, a valid FileRef
    assert _stored_hashes(artifacts) == {sha256_hex(b"o"), sha256_hex(b"e")}      # only the two log blobs
    # _stored_hashes: the sha256 keys that the test's artifact store holds

# tests/_agent/test_tool_run.py
async def test_reported_rows_recorded_as_tool_calls(...) -> None:
    # the tool writes 2 rows; the run has no gateway
    assert [c.role for c in _llm_calls(journal)] == ["tool", "tool"]   # _llm_calls: the LLMCall of each llm_call event (M3 shape)
    assert candidate.llm_calls_source == "reported" and len(candidate.llm_call_ids) == 2
    assert _settled_usd(journal, candidate) == (candidate.output.usage.usd or 0.0)   # rows never settle

async def test_oversize_llm_calls_file_warned_not_read(...) -> None:     # 17 MiB file
    assert "over the 16 MiB limit; not collected" in candidate.warnings[-1] and not _events(journal, "llm_call")

async def test_tool_with_providers_and_no_rows_is_unknown(...) -> None:
    assert candidate.llm_calls_source == "unknown"

# tests/_journal/test_llm_record.py
async def test_none_payload_not_stored(...) -> None:   # tracing.store_payloads is True
    call = await record_call(..., request=None, response=None, settle=False)
    assert call is not None and call.request_ref is None and call.response_ref is None

# tests/_planning/test_validate.py
def test_providers_without_gateway_warns() -> None:
    warnings = validate_plan(_plan(tool=FunctionTool("_fakes.tools:add", providers=["openai"])), RunConfig())
    assert any("declares providers ['openai'] but there is no llm_gateway" in w for w in warnings)
```

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/_tooling/test_llm_calls.py tests/_exec/test_job_steps.py tests/_agent/test_tool_run.py tests/_journal/test_llm_record.py tests/_planning/test_validate.py -v`
Expected: FAIL. `test_llm_calls.py` fails with `ModuleNotFoundError: No module named 'swarm_arena._tooling.llm_calls'`. The others fail with `AttributeError: ... 'llm_calls_file'` or with assertion errors.

- [ ] **Step 3: Implement**

- `parse_llm_calls`: split on `b"\n"`. Skip empty lines with no warning. For each other line, use `json.loads` and then `LLMCallRow.model_validate`. Any exception gives one warning for that line, until `MAX_LLM_CALL_WARNINGS` line warnings; after that, bad lines give no warning of their own, and the parser adds one final warning `f"swarm/llm_calls.jsonl: {k} more bad lines skipped"`. Stop after `MAX_LLM_CALL_ROWS` non-empty lines, valid or not (the row-cap warning text does not change). Fill `provider` after validation.
- `upload_outputs`: if `workspace / LLM_CALLS_PATH` exists, `sha, size = hash_regular_file(workspace, LLM_CALLS_PATH)` first. A file larger than `MAX_LLM_CALLS_BYTES` is not uploaded (so it never goes into a Ray worker's `MemoryStore`), and `llm_calls_file = FileRef(name="llm_calls.jsonl", path=LLM_CALLS_REF_PATH, sha256=sha, size=size)`. Otherwise `data = read_regular_file(workspace, LLM_CALLS_PATH)`, `sha = artifacts.put_bytes(data)`, and the same `FileRef` with `size=len(data)`. Never `artifacts.put_file`, and never `path=LLM_CALLS_PATH`: `FileRef.path` goes through `check_relpath`, which refuses the reserved `swarm/` (M1a Task 6). Both reads use the M1a `read_regular_file` rule, so symlinks are rejected, as for output files (M1a path rules).
- `execute_tool_run`, after `submit`: if `output.llm_calls_file` is set and its `size > MAX_LLM_CALLS_BYTES`, add the oversize warning and never fetch it. Otherwise `parse_llm_calls(artifacts.get_bytes(sha))`, add its warnings to `Candidate.warnings`, call `record_reported` with metadata `{run_id, goal_id, team, agent, round, role: "tool", candidate_id}`, and put the returned ids into `Candidate.llm_call_ids`. Then set `llm_calls_source` with `keyed=False`; Task 15 sets `keyed`, and Task 15 skips `record_reported` for keyed runs. `may_call` uses the same allowlist names that the M1b `secrets.allow` check reads.
- `record_call`: store a payload only when it is not `None`.
- `validate_plan`: add the warning for every agent and converge-team tool with non-empty `providers()` when `config.llm_gateway is None`.
- Docs: `docs/own-files.md` gets a section "Reporting LLM calls: `swarm/llm_calls.jsonl`". It gives the row fields above, one JSON object per line, `schema_version` optional, the 16 MiB and 10,000-row limits, the `SWARM_ARENA_*` variables (`SWARM_ARENA_RUN_ID` as the Langfuse `session_id`, `SWARM_ARENA_CANDIDATE_ID` as `metadata.candidate_id`), and the meaning of each `llm_calls_source` value. `docs/writing-a-connector.md` links to it. Add numpydoc docstrings with `Examples` on the new module's functions (private module: ruff `D` rules).

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/_tooling tests/_exec tests/_agent tests/_journal tests/_planning -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_tooling/llm_calls.py src/swarm_arena/_model/tooling.py src/swarm_arena/_exec/job_steps.py src/swarm_arena/_agent/tool_run.py src/swarm_arena/_journal/llm_record.py src/swarm_arena/_planning/validate.py tests docs
git commit -m "feat: collect swarm/llm_calls.jsonl from cooperating tools" -m "Rows become LLMCall events with role tool, recorded with settle=False, and the candidate's llm_calls_source says where its calls came from. ToolOutput gains the optional llm_calls_file, because the file is made where the tool runs and must reach the driver with the output (roadmap R4.2, R5.4 item 4, and an exception to the M4 Must NOT). validate_plan warns when a tool declares providers and there is no gateway."
```

---

### Task 4: Ray extra and object-store file transport

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies] ray`, ruff `TID251` per-file ignores), `uv.lock`
- Modify: `tests/conftest.py` (`collect_ignore` for the Ray test paths of R3.2 when `_missing("ray")`, the M1a helper that both conftest files define: `importlib.metadata.version("ray")` raises `PackageNotFoundError`). The root `conftest.py` already ignores `src/swarm_arena/ray` by the same `_missing("ray")` rule (M1a Task 1); keep it. Never use `importlib.util.find_spec("ray")`: `pythonpath = ["tests"]` makes it return the `tests/ray/` namespace package when Ray is not installed, so the `core` CI job would collect the Ray modules.
- Modify: `.github/workflows/ci.yml` (`extras` job: a step `uv run pytest -m "ray and not s3" -v` after the default pytest step; the M2 step `uv run pytest -m s3 -v` becomes `-m "s3 and not ray"`, so it never starts a Ray cluster)
- Create: `src/swarm_arena/ray/__init__.py`, `src/swarm_arena/ray/_files.py`
- Test: `tests/ray/test_files.py` (marker `ray`), `tests/test_import_isolation.py`

**Interfaces:**
- Consumes: `Store`, `Blob`, `AlreadyExists`, `NotFound` (R4.13.1, R4.13.2), `ArtifactStore`, `Job.Stage`, `ParentRef` (R4.7, R4.2).
- Produces:

```python
# ray/__init__.py
# MSG = "swarm_arena.ray needs the 'ray' extra: pip install 'swarm-arena[ray]'"
# try: importlib.metadata.version("ray")
# except importlib.metadata.PackageNotFoundError as e: raise ModuleNotFoundError(MSG, name="ray") from e
# try: import ray (and, from Task 7, the private-module imports)
# except ModuleNotFoundError as e:
#     if e.name == "ray": raise ModuleNotFoundError(MSG, name="ray") from e    # exact name, the M5 _EXTRA_DEPS rule
#     raise                                          # any other missing module, ray._private included: unchanged
#   Nothing broader is caught, so a plain ImportError (for example "cannot import name X from 'ray'") passes
#   through unchanged. The distribution check comes first: under pytest (`pythonpath = ["tests"]`) without Ray,
#   `import ray` succeeds and returns the `tests/ray/` namespace package. ModuleNotFoundError is an ImportError, and
#   pytest.importorskip("swarm_arena.ray") skips on it with no exc_type warning.
__all__: list[str] = []                              # Task 7 adds "RayExecutor"

# ray/_files.py
MIB: Final = 1024 * 1024
class MemoryStore(Store):                            # key = sha256 hex; used on the worker for a local artifact_root
    def __init__(self, blobs: Mapping[str, bytes] | None = None) -> None: ...
    def created(self) -> dict[str, bytes]: ...       # blobs created after __init__ (outputs and logs)
    # read, create (AlreadyExists), replace (if_match on sha256 etag), exists, list (sorted keys with the prefix)
def uses_object_store(artifact_root: str) -> bool: ...           # not artifact_root.startswith("s3://")
def needed_hashes(stage: Job.Stage) -> set[str]: ...             # root values, each parent's overlay values and file sha256s,
                                                                 # candidate files values
def inline_inputs(stage: Job.Stage, artifacts: ArtifactStore, *, cap_bytes: int) -> dict[str, bytes]: ...
def check_cap(blobs: Mapping[str, bytes], *, cap_bytes: int, what: Literal["inputs", "outputs"]) -> None: ...
def frame(head: bytes, blobs: Mapping[str, bytes]) -> bytes: ...
def unframe(data: bytes) -> tuple[bytes, dict[str, bytes]]: ...  # ValueError on a bad magic or a hash mismatch
```

Cap error text (exact, `ValueError`): `f"{what} of this tool run are {n} bytes, over the Ray object_store_cap_mib cap of {cap_mib} MiB; use an s3:// artifact_root"`. The cap applies to each tool run, to its inputs and to its outputs (output files plus logs) separately [plan].

Frame format [plan]: `b"SAF1"`, then the 8-byte big-endian length of `head`, then `head`. Then each blob in sorted sha256 order: the 64 ASCII hex chars of its sha256, the 8-byte big-endian length, and the bytes. No pickle.

pyproject `TID251` ignores: `"src/swarm_arena/ray/**"`, `"tests/ray/**"`, `"tests/e2e/test_rosenbrock_ray.py"`, `"tests/e2e/test_ray_cancel.py"`, `"tests/e2e/test_ray_dispatch_crash.py"`.

- [ ] **Step 1: Add the extra**

Run: `uv add --optional ray 'ray[default]>=2.59,<3'` and add the ruff per-file ignores.
Expected: `pyproject.toml` has `ray = ["ray[default]>=2.59,<3"]`, and `uv.lock` is updated.

- [ ] **Step 2: Write the failing tests**

```python
# tests/ray/test_files.py
pytestmark = pytest.mark.ray

def test_needed_hashes_cover_root_parents_and_candidate() -> None:
    stage = Job.Stage(root={"a.txt": "h1"}, parents=[ParentRef(candidate_id="c", run_id="r", tool_identity="t",
                      overlay={"b.txt": "h2"}, files=[FileRef(name="o", path="o", sha256="h3", size=1)])])
    assert needed_hashes(stage) == {"h1", "h2", "h3"}

def test_inline_inputs_over_cap_raises(tmp_path: Path) -> None:
    sha = artifacts.put_bytes(b"x" * (2 * MIB))
    with pytest.raises(ValueError, match="over the Ray object_store_cap_mib cap of 1 MiB"):
        inline_inputs(Job.Stage(root={"big.bin": sha}), artifacts, cap_bytes=1 * MIB)

def test_frame_roundtrip() -> None:
    blobs = {sha256_hex(b"a"): b"a", sha256_hex(b"bb"): b"bb"}
    assert unframe(frame(b'{"k": 1}', blobs)) == (b'{"k": 1}', blobs)

def test_unframe_rejects_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="sha256"):
        unframe(frame(b"{}", {"0" * 64: b"not that"}))

def test_memory_store_create_is_conditional() -> None:
    store = MemoryStore()
    store.create("k", b"1")
    with pytest.raises(AlreadyExists):
        store.create("k", b"1")
    assert store.created() == {"k": b"1"}

def test_uses_object_store() -> None:
    assert not uses_object_store("s3://bucket/a") and uses_object_store("/tmp/artifacts")

# tests/test_import_isolation.py
def test_ray_facade_error_without_extra() -> None:   # subprocess with sys.modules["ray"] = None
    result = subprocess.run([sys.executable, "-c", BLOCK_RAY + "import swarm_arena.ray"], capture_output=True, text=True)
    assert "pip install 'swarm-arena[ray]'" in result.stderr

def test_ray_facade_error_with_tests_dir_on_path() -> None:   # the pytest layout: tests/ray/ is a namespace package
    code = HIDE_RAY_DIST + "import sys; sys.path.insert(0, 'tests'); import swarm_arena.ray"   # repo root as cwd
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO_ROOT)
    assert "pip install 'swarm-arena[ray]'" in result.stderr and "ModuleNotFoundError" in result.stderr

def test_ray_facade_keeps_other_import_errors() -> None:
    result = subprocess.run([sys.executable, "-c", FAKE_RAY_DIST + BOOM_FINDER + "import swarm_arena.ray"],
                            capture_output=True, text=True)
    assert "boom inside ray" in result.stderr and "pip install" not in result.stderr

def test_ray_facade_keeps_missing_vendor_submodule() -> None:   # Ray is there, its own submodule is missing
    result = subprocess.run([sys.executable, "-c", FAKE_RAY_DIST + MISSING_SUB_FINDER + "import swarm_arena.ray"],
                            capture_output=True, text=True)
    assert "No module named 'ray._private'" in result.stderr and "swarm-arena[" not in result.stderr
```

`BLOCK_RAY` sets `sys.modules["ray"] = None`. `FAKE_RAY_DIST` patches `importlib.metadata.version` to return `"2.59.0"` for `"ray"`. `BOOM_FINDER` puts first on `sys.meta_path` a finder that raises `ImportError("boom inside ray")` for `"ray"`. `MISSING_SUB_FINDER` puts first on `sys.meta_path` a finder that raises `ModuleNotFoundError("No module named 'ray._private'", name="ray._private")` for `"ray"`, as `import ray` does when its own submodule is missing. A root-of-name check (`e.name.partition(".")[0] == "ray"`) fails this test. `HIDE_RAY_DIST` patches `importlib.metadata.version` so that `version("ray")` raises `PackageNotFoundError`, which is the `core` CI job's state even when the extras job has Ray installed. `REPO_ROOT = Path(__file__).resolve().parents[1]`.

- [ ] **Step 3: Run the tests to check that they fail**

Run: `uv run pytest tests/ray/test_files.py -m ray -v; uv run pytest tests/test_import_isolation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena.ray'`.

- [ ] **Step 4: Implement `ray/__init__.py` and `ray/_files.py` with the signatures above**

- [ ] **Step 5: Run the tests to check that they pass**

Run: `uv run pytest tests/ray/test_files.py -m ray -v && uv run pytest tests/test_import_isolation.py -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/conftest.py .github/workflows/ci.yml src/swarm_arena/ray tests/ray/test_files.py tests/test_import_isolation.py
git commit -m "feat: add the ray extra and object-store file transport" -m "With a local artifact_root, a Ray tool run's input and output files travel inside the task payload and result, each direction capped at object_store_cap_mib (256 MiB). The frame format is length-prefixed and hash-checked, with no pickle. The facade and tests/conftest.py detect Ray by its distribution (importlib.metadata), because under pytest the tests/ray/ directory is a namespace package that 'import ray' and find_spec('ray') would find. Only a missing ray module becomes the install hint; other import errors pass through unchanged."
```

---

### Task 5: The generic remote task

**Files:**
- Create: `src/swarm_arena/ray/_task.py`
- Create: `tests/ray/conftest.py` (`ray_driver` fixture; roadmap R3.2 gets this row)
- Modify: `src/swarm_arena/_exec/supervisor.py` (extract `launch_argv` from `ProcessSupervisor.run` step 1; behavior unchanged)
- Modify: `tests/conftest.py` (session fixture `ray_cluster`)
- Modify: `tests/_fakes/tools.py` (the scripts below)
- Modify: roadmap R4.13.16 (`job_payload` signature), R3.1 (`_exec/supervisor.py` `+M4`)
- Test: `tests/ray/test_task.py` (marker `ray`), `tests/_exec/test_supervisor.py`

**Interfaces:**
- Consumes: `prepare_job`, `resolve_argv`, `collect_output`, `upload_outputs`, `live_request` (Task 2); `MemoryStore`, `frame`, `unframe`, `check_cap`, `uses_object_store`, `inline_inputs` (Task 4); `put_edits`, `edit_hashes` (R4.13.9, tests only); `LocalExecutor` and the `local_executor` fixture (M1a, tests only); `ArtifactStore`, `open_store` (R4.13.2); `build_env` in `_exec/env.py`, `note_in_log` (M1a, `O_APPEND | O_NOFOLLOW | O_NONBLOCK`, regular files only, R4.13.8), `supervisor.BEFORE_KILL_TIMEOUT_S` (M1a, read at call time), `RELEASE` and `BlockingCollectTool` in `tests/_fakes/tools.py` (M1a Task 17), `RLimits` and the launcher CLI `python -I -m swarm_arena._launch --go-fd FD [--rlimit-as BYTES] [--rlimit-cpu SECONDS] [--rlimit-nofile N] -- ARGV...` (M1a, reused unchanged); `FunctionTool`, `import_root` (M1a `_tooling/function_tool.py`); `RunStatus`, `CompletedRun`, `ToolOutput`; Task 1 item 21 (a soft-cancelled task that catches `KeyboardInterrupt` returns its bytes normally through `ray.get`), which the soft-cancel return path of `run_in_worker` relies on.
- Produces:

```python
# _exec/supervisor.py (+M4)
def launch_argv(argv: Sequence[str], *, go_fd: int, rlimits: RLimits | None) -> list[str]: ...
    # [sys.executable, "-I", "-m", "swarm_arena._launch", "--go-fd", str(go_fd), <the M1a --rlimit-* flags>, "--", *argv];
    # ProcessSupervisor.run step 1 now calls it

# ray/_task.py
class TaskSettings(FrozenModel):
    poll_s: float = 0.2
    cap_bytes: int | None = None                 # set only when files travel through the object store
    import_roots: tuple[str, ...] = ()           # [plan] driver-side import roots (driver_import_roots)
def job_payload(job: Job, settings: TaskSettings, *, blobs: Mapping[str, bytes] | None = None) -> bytes: ...
    # frame(<Job JSON without the callables on_started and on_partial (never serialized), secret_values included>
    # + settings, blobs). Never logged or journaled. The request's edits travel as the R4.2 stored references
    # "sha256:<hex>", never as contents (the serializer and the reserved-form rule are unchanged).
def read_payload(payload: bytes) -> tuple[Job, TaskSettings, dict[str, bytes]]: ...
    # rebuilds the Job with on_started=None and on_partial=None; job.request.edits still holds the references
    # (run_in_worker step 2 restores the contents)
def driver_import_roots(job: Job) -> tuple[str, ...]: ...
    # runs on the driver: a kind="tool" job whose job.tool builds a FunctionTool → (import_root(<its function's
    # module>),); any other job → (). The driver has already imported that module (Plan.base_dir rule, or its own sys.path).
@ray.remote(max_retries=0, max_calls=1)
def run_job(payload: bytes) -> bytes: ...        # → frame(ToolOutput JSON, blobs created on the worker)
def run_in_worker(job: Job, workspace: Path, artifacts: ArtifactStore, *, poll_s: float,
                  import_roots: Sequence[str] = ()) -> ToolOutput: ...
    # the body of run_job without Ray, so most tests need no cluster

# tests/conftest.py
@pytest.fixture(scope="session")
def ray_cluster(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]: ...
    # runs <venv>/bin/ray start --head --port=<free> --num-cpus=4 --resources='{"llm_keys": 1}'
    # --disable-usage-stats, with PYTHONPATH=<repo>/tests so workers import _fakes, and, when
    # SWARM_ARENA_TEST_S3_ENDPOINT is set, AWS_ENDPOINT_URL_S3=<that endpoint> (Task 7 s3 test); sets RAY_ADDRESS for
    # the session; yields "127.0.0.1:<port>"; teardown: ray stop --force. It does not import ray.
# tests/ray/conftest.py
@pytest.fixture(scope="session")
def ray_driver(ray_cluster: str) -> Iterator[None]: ...
    # ray.init(address=ray_cluster, ignore_reinit_error=True); no ray.shutdown(), because RayExecutor tests in the
    # same pytest process may have called ray.init already, and other tests share the connection
```

`run_in_worker` steps (the tests do not fix the order, so it is pinned here):
1. Put each entry of `import_roots` (`TaskSettings.import_roots`, filled on the driver by `driver_import_roots`) at the front of `sys.path`. The worker exits after one task (`max_calls=1`), so the change ends with it. Then a `FunctionTool` function that only the driver's `sys.path` could import also loads on the worker, and its `command()` computes the same `--import-root` as on the driver. The path must exist on the worker's node (the same host, a shared file system, or the same image path); Task 10 documents this.
2. Restore the request, then prepare. When `job.request` is not `None`: `job = dataclasses.replace(job, request=live_request(job.request, artifacts))` (Task 2). A Job that crossed `job_payload` and `read_payload` holds each edit as its `"sha256:<hex>"` reference, so without this step `Tool.command` and `Tool.collect` would see references where `LocalExecutor` passes contents. Every reference resolves in a run: `put_edits` stores each edit before `plan_tool_stage` puts it in `stage.root` under the same hash (R4.13.10, M1a `check_tool`; edits are staged last, R4.13.9), so the object-store blobs (`needed_hashes`) and a shared `artifact_root` both hold it. A `ValueError` from `live_request` returns `ToolOutput(status=RunStatus.ERROR, data={"swarm_arena_error": str(e)})` at once: no `prepare_job`, no `command()` call, and no process. Then `command, tool = prepare_job(job, workspace, artifacts)`. Every later step uses this live `job`, the partial collect of step 7 and the `collect_output` of step 8 too.
3. Build the environment with the M1a `build_env`. The node environment is this worker's `os.environ`, and `secrets.file` is read here, on the worker. Then add `CUDA_VISIBLE_DEVICES` from `os.environ` when it is set.
4. Start the tool through the M1a launcher: `r, w = os.pipe()`; `p = subprocess.Popen(launch_argv(resolve_argv(command), go_fd=r, rlimits=RLimits(memory_bytes=int(memory_gib * 2**30) if memory_gib else None)), cwd=..., env=..., stdin=DEVNULL, stdout=<file>, stderr=<file>, pass_fds=(r,))`; close `r`; write `b"go"` to `w` and close it at once (the dispatch intent is already committed, so there is no per-process start event on Ray). Ignore a `BrokenPipeError` from the `b"go"` write: under `-I` the launcher can exit before "go" (for example, when `swarm_arena` is not importable in isolated mode). `p.wait()` then gives the exit code, so the status is `error` by the M1a rules, and the traceback is in the stderr log (R4.13.8 step 3, M1a Task 15 Step 3.3). `_launch` applies the rlimits in the child and then calls `execve`, so `p` is the tool process. Never pass `preexec_fn` (not safe in a multithreaded process such as a Ray worker), `start_new_session`, or `process_group`, so the tool stays in the worker's process group.
5. Loop on `p.wait(timeout=poll_s)`. When `job.timeout_s` passes (measured from `Popen`, so queue time does not count), stop with status `timeout`.
6. `except KeyboardInterrupt` (a soft `ray.cancel`): stop with status `cancelled`. The task then returns normally (Task 1 item 21).
7. Stop means `p.terminate()`, then poll until `job.grace_s` ends. Then, before the kill, as M1a `before_kill` does: run `collect_output(job, tool, workspace, <provisional CompletedRun(returncode=None, status=TIMEOUT or CANCELLED)>)` in a `threading.Thread(daemon=True)` whose target stores the output or the exception, and `join(supervisor.BEFORE_KILL_TIMEOUT_S)` (the module constant, read at call time, so `ray/_task.py` imports the `supervisor` module, not the name). The thread must be a daemon: `max_calls=1` ends the worker after the task, and a blocked collect must not keep it alive. A thread that is still alive after the join, or a stored exception, gives no partial. Then, in a `finally` (so a blocked or raising collect never skips the kill): `p.kill()` if `p` is still alive, then `p.wait()`. Only after the kill: `note_in_log(stderr, f"before_kill timed out after {n} s")`, or `note_in_log(stderr, f"before_kill failed: {type(e).__name__}: {e}")`, the M1a texts (`note_in_log` opens with `O_NONBLOCK` and drops a note to a non-regular file, so a FIFO log cannot block). Use the M1a status rules: `ok` or `error` by exit code, and `oom` when a SIGKILL or SIGSEGV that we did not send arrives while `memory_bytes` is set.
8. `upload_outputs(<output>, workspace, run, artifacts)`, with `run = CompletedRun(...)`. `<output>` is the stored partial after a stop; `ToolOutput(status=run.status)` after a stop whose collect timed out or raised, and `collect_output` is not tried again (the M1a Task 17 step 7 rule); else `collect_output(job, tool, workspace, run)`. This gives the partial output after a timeout or a cancel.

`run_job`: `read_payload`. Use `ArtifactStore(MemoryStore(blobs))` when `settings.cap_bytes` is set, else `ArtifactStore(open_store(job.artifact_root))`. Make a temporary workspace and call `run_in_worker(..., poll_s=settings.poll_s, import_roots=settings.import_roots)`. Then `check_cap(memory.created(), what="outputs")`. On a cap error, return `ToolOutput(status=RunStatus.ERROR, data={"swarm_arena_error": <message>})` with no blobs. Return `frame(output JSON, memory.created())`.

New scripts in `tests/_fakes/tools.py`: `partial_on_sigterm.py [--ready-file PATH]` (writes `{"data": {"step": n}}` atomically every 0.2 s, and on SIGTERM; writes `PATH` after its first step, so a test cancels only once the tool runs), `ignore_sigterm.py [--pid-file PATH] [--fifo-stderr]` (writes its pid to the absolute `PATH` when given; with `--fifo-stderr` it then replaces its stderr log path `swarm/stderr.log` with a FIFO, `os.remove` then `os.mkfifo`, and its open stderr stays the old file), `spawn_grandchild.py --pid-file PATH [--exit-after SECONDS] [--ignore-term]` (starts `sleep 300`, writes its pid to the absolute `PATH`, then sleeps; exits after `SECONDS` when given; ignores SIGTERM with `--ignore-term`), `report_env.py` (writes the selected env values, `os.getpgrp()`, and `os.getpgid(os.getppid())` into `data`). A module helper `_write_module(path: Path) -> None` in `tests/_fakes/tools.py` writes `def f(a: int) -> dict[str, int]: return {"a": a}` to `path`. New connector `EditSpyTool(PyTool)` in `tests/_fakes/tools.py` records the edits that each method sees, in the process where it runs (the Ray worker too): `command(request, workspace)` writes `json.dumps(request.edits)` to `workspace / "edit_spy.json"`, then returns `super().command(request, workspace)`; `collect(request, workspace, run)` returns `ToolOutput(status=run.status, data={"command_edits": <edit_spy.json parsed>, "collect_edits": dict(request.edits)})`, so a test sees whether the final collect (`ok`) or the partial collect before the kill (the provisional `timeout`) made it.

- [ ] **Step 1: Write the failing tests**

```python
pytestmark = pytest.mark.ray

def test_run_in_worker_ok(tmp_path: Path) -> None:
    assert run_in_worker(_job("report_env.py"), ws, artifacts, poll_s=0.2).status == RunStatus.OK

def test_timeout_keeps_partial_output(tmp_path: Path) -> None:
    out = run_in_worker(_job("partial_on_sigterm.py", timeout_s=1.0, grace_s=2.0), ws, artifacts, poll_s=0.2)
    assert out.status == RunStatus.TIMEOUT and out.partial and out.data["step"] >= 1

def test_tool_ignoring_sigterm_killed_after_grace(tmp_path: Path) -> None:
    start = time.monotonic()
    out = run_in_worker(_job("ignore_sigterm.py", timeout_s=1.0, grace_s=1.0), ws, artifacts, poll_s=0.2)
    assert out.status == RunStatus.TIMEOUT and time.monotonic() - start < 1.0 + 1.0 + 2.0

def test_blocked_collect_cannot_stop_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:   # no cluster
    monkeypatch.setattr(supervisor, "BEFORE_KILL_TIMEOUT_S", 0.5)
    code = f"import os, time; open({str(tmp_path / 'pid')!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    try:
        start = time.monotonic()
        out = run_in_worker(_blocking_job(code, timeout_s=1.0, grace_s=0.2), ws, artifacts, poll_s=0.2)
        assert (out.status, out.partial) == (RunStatus.TIMEOUT, False) and time.monotonic() - start < 10.0
        _wait_until(lambda: not _alive(int((tmp_path / "pid").read_text())), timeout_s=5.0)
    finally:
        RELEASE.set()                                                    # frees the daemon collect thread

def test_fifo_stderr_cannot_block_kill(tmp_path: Path) -> None:          # no cluster
    pid_file = tmp_path / "pid"
    start = time.monotonic()
    out = run_in_worker(_job("ignore_sigterm.py", args=["--pid-file", str(pid_file), "--fifo-stderr"], timeout_s=1.0,
                             grace_s=0.5), ws, artifacts, poll_s=0.2)
    assert out.status == RunStatus.TIMEOUT and time.monotonic() - start < 10.0
    _wait_until(lambda: not _alive(int(pid_file.read_text())), timeout_s=5.0)

def test_swarm_arena_env_and_cuda_reach_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    out = run_in_worker(_job("report_env.py", env={"SWARM_ARENA_CANDIDATE_ID": "c-x-e1-0001"}), ws, artifacts, poll_s=0.2)
    assert out.data["env"]["SWARM_ARENA_CANDIDATE_ID"] == "c-x-e1-0001" and out.data["env"]["CUDA_VISIBLE_DEVICES"] == "1"

def test_secrets_read_on_worker_never_in_payload(tmp_path: Path) -> None:
    (tmp_path / "secrets.env").write_text("SA_TEST_SECRET=from-file\n")
    job = _job("report_env.py", secret_env=("SA_TEST_SECRET",), secrets_file=str(tmp_path / "secrets.env"))
    assert run_in_worker(job, ws, artifacts, poll_s=0.2).data["env"]["SA_TEST_SECRET"] == "from-file"
    assert b"from-file" not in job_payload(job, TaskSettings())

def test_import_roots_make_function_importable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", list(sys.path))                   # run_in_worker edits sys.path of this process
    _write_module(tmp_path / "mods" / "modx_w.py")                      # not on this process's sys.path
    out = run_in_worker(_function_job("modx_w:f", inputs={"a": 1}), ws, artifacts, poll_s=0.2,
                        import_roots=[str(tmp_path / "mods")])
    assert out.status == RunStatus.OK and out.data == {"a": 1}

def test_driver_import_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_module(tmp_path / "mods" / "modx_d.py")
    monkeypatch.syspath_prepend(str(tmp_path / "mods"))                 # the driver imports it, as the Plan.base_dir rule does
    assert driver_import_roots(_function_job("modx_d:f")) == (str((tmp_path / "mods").resolve()),)
    assert driver_import_roots(_job("report_env.py")) == ()

def test_run_job_options() -> None:
    assert run_job._default_options["max_retries"] == 0 and run_job._default_options["max_calls"] == 1

def test_soft_cancel_returns_partial_through_ray(ray_driver: None, tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    job = _job("partial_on_sigterm.py", args=["--ready-file", str(ready)], grace_s=2.0)
    ref = run_job.remote(job_payload(job, TaskSettings(cap_bytes=256 * MIB)))
    _wait_until(ready.exists, timeout_s=30.0)                       # polls every 0.05 s
    ray.cancel(ref)
    out = ToolOutput.model_validate_json(unframe(ray.get(ref))[0])
    assert out.status == RunStatus.CANCELLED and out.partial and out.data["step"] >= 1

def test_tool_in_worker_process_group(ray_driver: None) -> None:
    data = _remote_data(_job("report_env.py"))
    assert data["pgid"] == data["parent_pgid"]

def test_grandchild_cleaned_after_finish(ray_driver: None, tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    _remote_data(_job("spawn_grandchild.py", args=["--pid-file", str(pid_file), "--exit-after", "0.5"]))
    pid = int(pid_file.read_text())
    _wait_until(lambda: not _alive(pid), timeout_s=5.0)

def test_output_over_cap_is_error(ray_driver: None) -> None:        # cap 1 MiB, tool writes a 2 MiB output file
    out = ToolOutput.model_validate_json(unframe(ray.get(run_job.remote(_payload_with_cap(1 * MIB))))[0])
    assert out.status == RunStatus.ERROR and "object_store_cap_mib" in out.data["swarm_arena_error"]

def test_payload_has_no_callables() -> None:
    job, _, _ = read_payload(job_payload(_job("report_env.py", on_started=_noop, on_partial=_noop), TaskSettings()))
    assert job.on_started is None and job.on_partial is None

EDITS = {"p.txt": "hello", "q.txt": "sha256:hello"}      # ordinary text, and content that only looks like a reference

@pytest.mark.parametrize("object_store", [True, False])   # blobs in the payload, or the shared artifact_root
@pytest.mark.parametrize(("code", "timeout_s", "status"), [
    ("pass", None, RunStatus.OK),                              # normal exit: the step 8 collect
    ("import time; time.sleep(60)", 1.0, RunStatus.TIMEOUT),   # stop: the step 7 partial collect, before the kill
])
async def test_edit_contents_same_on_ray_as_local(ray_driver: None, local_executor: LocalExecutor, tmp_path: Path,
                                                  object_store: bool, code: str, timeout_s: float | None,
                                                  status: RunStatus) -> None:
    job, artifacts = _edit_spy_job(tmp_path, EDITS, code=code, timeout_s=timeout_s)
    blobs = inline_inputs(job.stage, artifacts, cap_bytes=256 * MIB) if object_store else None
    payload = job_payload(job, TaskSettings(cap_bytes=256 * MIB if object_store else None), blobs=blobs)
    assert read_payload(payload)[0].request.edits == {p: f"sha256:{h}" for p, h in edit_hashes(EDITS).items()}
    remote = ToolOutput.model_validate_json(unframe(await anyio.to_thread.run_sync(ray.get, run_job.remote(payload)))[0])
    local = await local_executor.submit(job)
    assert remote.status == local.status == status                    # EditSpyTool.collect returns run.status
    assert remote.data == local.data == {"command_edits": EDITS, "collect_edits": EDITS}

def test_unrestorable_edit_reference_never_starts_tool(tmp_path: Path) -> None:   # no cluster
    job, _ = _edit_spy_job(tmp_path, {"p.txt": "hello"})
    sent, _, _ = read_payload(job_payload(job, TaskSettings()))
    out = run_in_worker(sent, ws, ArtifactStore(MemoryStore()), poll_s=0.2)          # the edit's blob is not there
    assert out.status == RunStatus.ERROR and out.data["swarm_arena_error"].startswith("edit 'p.txt': stored reference")
    assert not (ws / "edit_spy.json").exists()                                        # command() never ran

# tests/_exec/test_supervisor.py
def test_launch_argv() -> None:
    assert launch_argv(["x"], go_fd=5, rlimits=RLimits(memory_bytes=1024, nofile=64)) == [
        sys.executable, "-I", "-m", "swarm_arena._launch", "--go-fd", "5", "--rlimit-as", "1024", "--rlimit-nofile", "64",
        "--", "x"]
```

`_blocking_job(code, **kw)` builds a kind `"tool"` job with `BlockingCollectTool` (M1a Task 17, its `collect` first calls `RELEASE.wait(30)`) running the `PyTool` code `code`. `supervisor` is the module `swarm_arena._exec.supervisor`. `_edit_spy_job(tmp_path, edits, *, code="pass", timeout_s=None)` returns `(job, artifacts)`: `artifacts = ArtifactStore(open_store(str(tmp_path / "artifacts")))`, `put_edits(edits, artifacts)`, then a kind `"tool"` job with `Ref(path="_fakes.tools:EditSpyTool", config={"code": code})`, `request=ToolRequest(edits=edits)`, `stage=Job.Stage(root=edit_hashes(edits))`, `artifact_root=str(tmp_path / "artifacts")`, `timeout_s=timeout_s`, and `grace_s=0.5`, staged as `execute_tool_run` stages edits. The timeout case covers the cancel path too: step 7 runs the same partial collect, with the same live `job`, for `timeout` and `cancelled`.

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/ray/test_task.py -m ray -v; uv run pytest tests/_exec/test_supervisor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena.ray._task'` and `ImportError: cannot import name 'launch_argv'`.

- [ ] **Step 3: Implement `launch_argv`, `ray/_task.py`, the two fixtures, and the scripts, as specified above**

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/ray/test_task.py tests/ray/test_files.py -m ray -v && uv run pytest tests/_exec -v && uv run ruff check . && uv run ty check`
Expected: PASS. All of `tests/_exec/test_supervisor.py` and `tests/_exec/test_local.py` stay green.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/ray/_task.py src/swarm_arena/_exec/supervisor.py tests/conftest.py tests/ray tests/_exec/test_supervisor.py tests/_fakes/tools.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat: add the generic Ray task that runs a tool in the worker's process group" -m "max_retries=0 and max_calls=1; the tool starts through the M1a launcher, which applies the rlimits (no preexec_fn, no setsid), and is polled with p.wait(timeout=0.2) so a soft cancel gets in; SIGTERM, grace_s, partial collect, kill. The partial collect runs in a daemon thread joined for BEFORE_KILL_TIMEOUT_S, so a blocked collect cannot delay the kill, and a FIFO log cannot block it. A FunctionTool's import root comes from the driver in the payload. Before prepare_job, the worker turns the request's sha256 edit references back into contents through the artifact store (live_request), so Tool.command and Tool.collect (the partial collect after a stop too) see the same request on Ray as on LocalExecutor; the stored references and the reserved-form rule stay. Secrets are read on the worker. Roadmap R4.13.16: job_payload takes TaskSettings and blobs."
```

---
### Task 6: Find a Ray task by name

**Files:**
- Create: `src/swarm_arena/ray/_state.py`
- Test: `tests/ray/test_state.py` (marker `ray`)

**Interfaces:**
- Consumes: `run_job`, `job_payload`, `TaskSettings` (Task 5); the Ray state names from Task 1 items 14 and 15.
- Produces:

```python
# ray/_state.py
PENDING_STATES: Final[frozenset[str]]       # Task 1 item 14: every state before the task runs
TERMINAL_STATES: Final[frozenset[str]]      # Task 1 item 14: expected {"FINISHED", "FAILED"}
class TaskInfo(FrozenModel):
    task_id: str
    name: str
    state: str
    worker_id: str | None
    node_id: str | None
def find_task(task_name: str) -> TaskInfo | None: ...
    # ray.util.state.list_tasks(filters=[("name", "=", task_name)], detail=True); the newest attempt; None if none
def worker_dead(task_name: str) -> bool | None: ...
    # None: no task found, or a terminal task whose worker record is gone. False: the task is not terminal,
    # or its worker is alive. True: the task is terminal, and it never got a worker (it never ran), or its worker's
    # is_alive is false.
```

Both functions are blocking. Async callers run them with `anyio.to_thread.run_sync`.

- [ ] **Step 1: Write the failing tests**

```python
pytestmark = pytest.mark.ray

def test_find_task_by_name(ray_driver: None) -> None:
    ref = run_job.options(name="x-test-e1-0001").remote(_payload("report_env.py"))
    ray.get(ref)
    info = _wait_for(lambda: find_task("x-test-e1-0001"), timeout_s=10.0)
    assert info.name == "x-test-e1-0001" and info.state in TERMINAL_STATES

def test_find_task_unknown_name_is_none(ray_driver: None) -> None:
    assert find_task("x-never-e1-0001") is None

def test_worker_not_dead_while_running(ray_driver: None) -> None:
    ref = run_job.options(name="x-test-e1-0002").remote(_payload("ignore_sigterm.py"))
    try:
        _wait_for(lambda: (i := find_task("x-test-e1-0002")) and i.state not in PENDING_STATES, timeout_s=10.0)
        assert worker_dead("x-test-e1-0002") is False
    finally:
        ray.cancel(ref, force=True)

def test_worker_dead_after_finish(ray_driver: None) -> None:            # max_calls=1: the worker exits
    ray.get(run_job.options(name="x-test-e1-0003").remote(_payload("report_env.py")))
    _wait_for(lambda: worker_dead("x-test-e1-0003") is True, timeout_s=10.0)

def test_worker_dead_after_force_cancel(ray_driver: None) -> None:
    ref = run_job.options(name="x-test-e1-0004").remote(_payload("ignore_sigterm.py"))
    try:
        time.sleep(1.0)
    finally:
        ray.cancel(ref, force=True)
    _wait_for(lambda: worker_dead("x-test-e1-0004") is True, timeout_s=10.0)
```

`_payload(script)` builds the job with `timeout_s=10.0`. With the `finally` cancels, no `ignore_sigterm.py` task still holds a CPU of the 4-CPU session cluster when Task 7's `Resources(cpus=4)` test runs.

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/ray/test_state.py -m ray -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena.ray._state'`.

- [ ] **Step 3: Implement `ray/_state.py` with the signatures and the Task 1 names**

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/ray/test_state.py -m ray -v && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/ray/_state.py tests/ray/test_state.py
git commit -m "feat: look up Ray tasks by name through ray.util.state" -m "find_task and worker_dead let resume and reconcile check a tool run from its committed dispatch-intent event, whose job id is the Ray task name."
```

---

### Task 7: `RayExecutor`

**Files:**
- Create: `src/swarm_arena/ray/_executor.py`
- Modify: `src/swarm_arena/ray/__init__.py` (`__all__ = ["RayExecutor"]`, then the last line `_publish(__name__, __all__)`, with `from swarm_arena._facade import publish as _publish` (M1a `_facade.publish`, R0). Do not set `__module__` by hand: `publish` also sets it on the methods, so autodoc and `--doctest-modules` find the method examples under `swarm_arena.ray`)
- Modify: `src/swarm_arena/_exec/executor.py` (private `schedule_timeout_scope` and `current_schedule_timeout`)
- Modify: `src/swarm_arena/_engine/run.py` and `src/swarm_arena/_engine/resume.py` (enter `schedule_timeout_scope(config.schedule_timeout_s)` around the engine)
- Create: `docs/api/ray.md` (`automodule:: swarm_arena.ray`, `:members:`)
- Modify: `docs/api/index.md` (add `ray` to its toctree of module pages, so `sphinx-build -W` finds the page in a toctree), `tests/test_public_api.py` (`swarm_arena.ray` snapshot, `pytest.importorskip("swarm_arena.ray")`), `docs/running-on-ray.md` (section "The Ray executor"), `CHANGELOG.md`, roadmap (M4 exit criteria command, R3.1 tags)
- Modify: `tests/_fakes/tools.py` (`StallCollectTool(PyTool)`: its `command()` runs `spawn_grandchild.py --pid-file <inputs["pid_file"]> --ignore-term`, and its `collect()` sleeps 60 s)
- Test: `tests/ray/test_executor.py`, `tests/e2e/test_ray_cancel.py` (both marker `ray`), `tests/_exec/test_executor.py`, `tests/_engine/test_run.py`

**Interfaces:**
- Consumes: `Executor`, `Job`, `Job.Started`, `Job.on_partial` (R4.7, M1a: the only channel by which an executor hands a cancel-path partial output to the caller); `run_job`, `job_payload`, `TaskSettings`, `driver_import_roots` (Task 5); `find_task`, `worker_dead`, `PENDING_STATES` (Task 6); `inline_inputs`, `uses_object_store`, `unframe`, `MIB` (Task 4); `ArtifactStore`, `open_store`; `check_tool` (R4.12, tests only); the `ray.nodes()` fields from Task 1 item 20; Task 1 item 21 for the soft-cancel return path (the ref of a soft-cancelled task gives the partial `ToolOutput`, not `TaskCancelledError`).
- Produces (R4.10, public):

```python
class RayExecutor(Executor):
    class Config(BaseModel):
        address: str = "auto"
        poll_s: float = 0.2
        cancel_margin_s: float = 10.0
        object_store_cap_mib: int = 256
        secret_resource: str | None = None
    def __init__(self, config: "RayExecutor.Config | None" = None, /, **fields: Any) -> None: ...   # no I/O
    def preflight(self) -> None: ...
    async def submit(self, job: Job) -> ToolOutput: ...
    async def capacity(self) -> Resources | None: ...
    async def aclose(self) -> None: ...
    async def confirm_dead(self, started: Job.Started) -> bool | None: ...
    async def kill_orphan(self, started: Job.Started) -> bool: ...
def task_options(job: Job, config: RayExecutor.Config) -> dict[str, Any]: ...   # private, pure
def request_shape(options: Mapping[str, Any]) -> dict[str, float]: ...           # private, pure
    # {"CPU": num_cpus, "GPU": num_gpus, "memory": memory, **resources}, zero and missing amounts left out
def fits_any_node(request: Mapping[str, float], nodes: Sequence[Mapping[str, float]]) -> bool: ...   # private, pure
    # True when some node's total has node.get(k, 0.0) >= v for every (k, v) in request

# _exec/executor.py (+M4), private context-variable seam [plan]
def schedule_timeout_scope(seconds: float | None) -> AbstractContextManager[None]: ...
def current_schedule_timeout() -> float | None: ...      # None outside any scope
# Swarm.run and Swarm.resume enter schedule_timeout_scope(config.schedule_timeout_s) before they start any task, so
# every tool run and every check or judge ctx.execute job sees it. LocalExecutor ignores it.
```

Pinned behavior:
- `preflight()`: a non-POSIX platform raises `RuntimeError`. An `address` that starts with `"ray://"` raises `ValueError("Ray Client (ray://) is not used for long runs; run the driver as a Ray Job with swarm-arena submit")`.
- The first call of `submit`, `capacity`, `confirm_dead`, or `kill_orphan` runs `ray.init(address=config.address, ignore_reinit_error=True)` once, under the `anyio.Lock`, through `anyio.to_thread.run_sync`. `_executor.py` calls `worker_dead` through its module global. `aclose()` never calls `ray.shutdown()` [plan], because the process may share Ray with other code.
- `task_options`: `name=job.id`, `num_cpus=resources.cpus`, `num_gpus=resources.gpus`, and `memory=int(memory_gib * 1024**3)` when `memory_gib` is set. `resources = {**resources.custom}`, plus `{secret_resource: 1}` when `secret_resource` is set and `job.secret_env` is not empty. There is never a `runtime_env` key.
- `submit` order:
  1. Init.
  2. If `uses_object_store(job.artifact_root)`: `inline_inputs(..., cap_bytes=object_store_cap_mib * MIB)`. A `ValueError` returns `ToolOutput(status=ERROR, data={"swarm_arena_error": <message>})`, and `on_started` is never called.
  3. If `job.on_started is not None`: `await job.on_started(Job.Started(kind="dispatch", job_id=job.id, task_name=job.id))`. An exception propagates, and nothing is submitted. A job without `on_started` (M1a `check_tool` jobs, and `CheckContext.execute` and `JudgeContext.execute` command jobs whose context has `on_started=None`, as in the conformance kit) is submitted with no intent event. In a run the core sets `on_started` on command jobs too, and M1b's callback journals the dispatch intent as `command_start` (R4.14).
  4. `run_job.options(**task_options(...)).remote(job_payload(job, TaskSettings(poll_s=..., cap_bytes=..., import_roots=driver_import_roots(job)), blobs=...))`. `_executor.py` calls `run_job` through its module global, so a test can replace it.
  5. In one task group: await the ref, and watch scheduling. The child that awaits the ref catches `ray.exceptions.RayError` itself (the Ray error rule below), stores that output, and cancels the group. The watch child stores the schedule-timeout output in the same way. No exception except cancellation leaves the task group, because anyio 4 wraps child errors in an `ExceptionGroup`. Every 1.0 s [plan], while `find_task(job.id)` is `None` (not reported yet) or its state is in `PENDING_STATES`, the watch reads the alive nodes' total resources (`ray.nodes()` in a thread) and calls `fits_any_node(request_shape(options), totals)`. The clock starts when the request first fits no node, and it resets when the request fits a node again (for example, a node joined). When the request has fit no node for `t = current_schedule_timeout()` seconds (`600.0`, the `RunConfig` default, when it is `None`), the watch child stores `ToolOutput(status=ERROR, data={"swarm_arena_error": f"no node met the resource request within schedule_timeout_s={t:g}"})` and cancels the group; after the group, `submit` runs the cancel steps on the ref (shielded: `ray.cancel(ref)`, the bounded wait, `ray.cancel(ref, force=True)`) and returns that output, with no `on_partial` call and no re-raise. A request that some node can meet waits for busy resources with no limit from this watch: queue time counts only toward `round_timeout_s` (§6.1).
  6. `unframe` the result, put every returned blob into `ArtifactStore(open_store(job.artifact_root))` (the hash is checked), and return the `ToolOutput`.
- Cancel: an `except BaseException:` around the whole task group of step 5 runs this sequence, in `anyio.CancelScope(shield=True)`: `ray.cancel(ref)`, then `with anyio.move_on_after(job.grace_s + cancel_margin_s)`: await the ref to get the partial output. Then `ray.cancel(ref, force=True)`, which does nothing if the task has already finished. Deliver the partial output exactly as `LocalExecutor` does on cancel (M1a): `await job.on_partial(output)` when it is set, where `output` is the result after step 6 (unframed, blobs stored), or `ToolOutput(status=CANCELLED)` when the wait ran out or the ref raised. `on_partial` is the only delivery path; M1b's `execute_tool_run` sets it. Then re-raise.
- A Ray error that we did not cause, caught by the child that awaits the ref: `ray.exceptions.OutOfMemoryError` → `OOM`; any other `ray.exceptions.RayError` → `ERROR`; both with `data={"swarm_arena_error": f"{type(e).__name__}: {e}"}`. The job is never resubmitted.
- `capacity()`: `ray.cluster_resources()` → `Resources(cpus=CPU, gpus=GPU, memory_gib=memory / 1024**3, custom=<the rest>)`. The rest leaves out `memory`, `object_store_memory`, `node:*`, and `accelerator_type:*`.
- `confirm_dead`: `kind == "dispatch"` → `worker_dead(started.task_name)` in a thread; otherwise `None`.
- `kill_orphan`: `kind == "dispatch"`. Ray kills the tasks of a dead driver (owner fate sharing, Task 1 item 16). This method polls `worker_dead` every 1.0 s for at most `cancel_margin_s`, and returns `True` only when the worker is confirmed dead. It returns `False` if the time runs out or the kind is not `"dispatch"`. It never kills anything itself.
- Docstrings: numpydoc with `Examples` on the class and on each public method. Use `# doctest: +SKIP` for anything that needs a cluster.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ray/test_executor.py
pytestmark = pytest.mark.ray

def test_preflight_rejects_ray_client_address() -> None:
    with pytest.raises(ValueError, match="Ray Client"):
        RayExecutor(address="ray://head:10001").preflight()

def test_task_options_maps_resources() -> None:
    job = _job(resources=Resources(cpus=2, gpus=1, memory_gib=1.5, custom={"deepretro": 1}))
    assert task_options(job, RayExecutor.Config()) == {"name": job.id, "num_cpus": 2, "num_gpus": 1,
                                                       "memory": 1_610_612_736, "resources": {"deepretro": 1}}

def test_task_options_adds_secret_resource_for_secret_env() -> None:
    opts = task_options(_job(secret_env=("ANTHROPIC_API_KEY",)), RayExecutor.Config(secret_resource="llm_keys"))
    assert opts["resources"] == {"llm_keys": 1} and "runtime_env" not in opts

async def test_init_runs_in_a_worker_thread(ray_cluster: str, monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[bool] = []
    monkeypatch.setattr(ray, "init", lambda **kw: threads.append(threading.current_thread() is threading.main_thread()))
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"CPU": 1.0})
    await RayExecutor().capacity()
    assert threads == [False]

async def test_dispatch_intent_before_submit(ray_cluster: str, monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    seen: list[Job.Started] = []
    monkeypatch.setattr("swarm_arena.ray._executor.run_job", _SpyRunJob(order))   # .options(**kw).remote(p): records
    async def on_started(s: Job.Started) -> None:                                   # "remote", then calls the real run_job
        order.append("on_started")
        seen.append(s)
    await RayExecutor().submit(_job("report_env.py", on_started=on_started))
    assert order == ["on_started", "remote"]
    assert seen[0] == Job.Started(kind="dispatch", job_id=seen[0].job_id, task_name=seen[0].job_id)

async def test_job_without_on_started_is_submitted(ray_cluster: str) -> None:
    executor = RayExecutor()
    tool_out = await executor.submit(_job("report_env.py"))                  # on_started=None
    command_out = await executor.submit(_sleep_job(0.1))                     # a command job, on_started=None
    assert tool_out.status == command_out.status == RunStatus.OK

async def test_kill_orphan_inits_ray_first(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ray, "init", lambda **kw: calls.append("init"))
    monkeypatch.setattr("swarm_arena.ray._executor.worker_dead", lambda name: calls.append("state") or True)
    started = Job.Started(kind="dispatch", job_id="x-t-e1-0400", task_name="x-t-e1-0400")
    assert await RayExecutor().kill_orphan(started) is True
    assert calls == ["init", "state"]

async def test_worker_crash_maps_to_error(ray_cluster: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("swarm_arena.ray._executor.run_job", _crash_run_job)   # max_retries=0; os._exit(1)
    out = await RayExecutor().submit(_job("report_env.py"))
    assert out.status == RunStatus.ERROR and out.data["swarm_arena_error"].startswith("WorkerCrashedError")

async def test_on_started_failure_submits_nothing(ray_cluster: str) -> None:
    with pytest.raises(RuntimeError, match="commit failed"):
        await RayExecutor().submit(_job("report_env.py", on_started=_raise("commit failed"), id="x-t-e1-0100"))
    await anyio.sleep(2.0)
    assert find_task("x-t-e1-0100") is None

async def test_submit_roundtrips_files_through_object_store(ray_cluster: str, tmp_path: Path) -> None:
    out = await RayExecutor().submit(_job("write_output.py", artifact_root=str(tmp_path / "art")))
    assert out.status == RunStatus.OK and _artifacts(tmp_path / "art").get_bytes(out.files[0].sha256) == b"result"

@pytest.mark.s3
async def test_s3_artifact_root_sends_only_hashes(ray_cluster: str, s3_url: str) -> None:   # the M2 s3 fixture
    if not os.environ.get("SWARM_ARENA_TEST_S3_ENDPOINT"):   # s3_url only reads it; see the note after this block
        pytest.skip("needs SWARM_ARENA_TEST_S3_ENDPOINT and AWS_* set before the session, so Ray workers reach the same S3")
    payload_sizes = _spy_payload_sizes()
    out = await RayExecutor().submit(_job("read_input.py", artifact_root=s3_url, input_bytes=1 * MIB))
    assert out.status == RunStatus.OK and max(payload_sizes) < 64 * 1024

async def test_inputs_over_cap_never_dispatched(ray_cluster: str) -> None:
    started: list[Job.Started] = []
    out = await RayExecutor(object_store_cap_mib=1).submit(_job(input_bytes=2 * MIB, on_started=_record(started)))
    assert out.status == RunStatus.ERROR and "object_store_cap_mib" in out.data["swarm_arena_error"] and started == []

async def test_schedule_timeout_fails_run(ray_cluster: str) -> None:
    start = time.monotonic()
    with schedule_timeout_scope(2.0):
        out = await RayExecutor().submit(_job(resources=Resources(custom={"no_such_node": 1})))
    assert out.status == RunStatus.ERROR and "schedule_timeout_s=2" in out.data["swarm_arena_error"]
    assert time.monotonic() - start < 2.0 + 1.0 + 10.0

async def test_feasible_task_queued_past_schedule_timeout_still_runs(ray_cluster: str) -> None:
    executor = RayExecutor()
    with schedule_timeout_scope(1.0):
        async with anyio.create_task_group() as tg:
            tg.start_soon(executor.submit, _sleep_job(4.0, resources=Resources(cpus=4)))   # holds every CPU for 4 s
            await anyio.sleep(1.0)
            start = time.monotonic()
            out = await executor.submit(_job("report_env.py", resources=Resources(cpus=1)))
    assert out.status == RunStatus.OK and time.monotonic() - start > 1.0 + 1.0   # it queued past the timeout, then ran

def test_fits_any_node() -> None:
    nodes = [{"CPU": 4.0, "memory": 8e9, "llm_keys": 1.0}]
    assert fits_any_node({"CPU": 4.0, "llm_keys": 1.0}, nodes) and not fits_any_node({"CPU": 5.0}, nodes)
    assert not fits_any_node({"no_such_node": 1.0}, nodes) and not fits_any_node({"CPU": 1.0}, [])

async def test_cancel_while_pending_never_runs_tool(ray_cluster: str, tmp_path: Path) -> None:   # Review Focus 2
    marker = tmp_path / "ran"
    start = time.monotonic()
    with anyio.move_on_after(0.5):
        await RayExecutor(cancel_margin_s=2.0).submit(_job("touch_marker.py", marker=marker, grace_s=1.0,
                                                           resources=Resources(custom={"no_such_node": 1})))
    assert time.monotonic() - start < 0.5 + 1.0 + 2.0 + 1.0
    await anyio.sleep(3.0)
    assert not marker.exists() and find_task(_last_job_id()).state in TERMINAL_STATES

async def test_cancel_partial_matches_local_executor(ray_cluster: str, tmp_path: Path) -> None:
    local, _ = await _cancel_and_capture(LocalExecutor(), _partial_job(tmp_path / "l"), ready=(tmp_path / "l").exists)
    remote, _ = await _cancel_and_capture(RayExecutor(), _partial_job(tmp_path / "r"), ready=(tmp_path / "r").exists)
    assert (local.status, local.partial) == (remote.status, remote.partial) == (RunStatus.CANCELLED, True)
    # _partial_job(ready_file): _job("partial_on_sigterm.py", args=["--ready-file", str(ready_file)], grace_s=2.0)
    # _cancel_and_capture(executor, job, *, ready: Callable[[], bool]) -> tuple[ToolOutput, float]: sets on_partial on
    # the Job (dataclasses.replace(job, on_partial=<async def that appends to got>); Job is frozen), submits it in
    # `with anyio.CancelScope() as scope:`, cancels the scope once ready() is true (polled with anyio.sleep(0.05);
    # pytest.fail after 30 s), and returns got[0], the output that the executor handed to Job.on_partial (M1a, R4.7),
    # and the seconds from the cancel to the return of submit

async def test_core_built_function_job_imports_on_worker(ray_cluster: str, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    _write_module(tmp_path / "mods" / "modx_c.py")
    monkeypatch.syspath_prepend(str(tmp_path / "mods"))              # only the driver's sys.path has it
    out = await check_tool(FunctionTool("modx_c:f"), {"a": 1}, executor=RayExecutor())   # the core builds the Job
    assert out.status == RunStatus.OK and out.data == {"a": 1}

async def test_secret_values_never_in_task_metadata(ray_cluster: str) -> None:
    await RayExecutor().submit(_job("report_env.py", id="x-t-e1-0200", secret_values={"OPENAI_API_KEY": "sk-virt-XYZ"}))
    info = ray.util.state.list_tasks(filters=[("name", "=", "x-t-e1-0200")], detail=True)[0]
    assert "sk-virt-XYZ" not in repr(info)

async def test_capacity_reports_cluster(ray_cluster: str) -> None:
    cap = await RayExecutor().capacity()
    assert cap is not None and cap.cpus == 4.0 and cap.custom == {"llm_keys": 1.0}

async def test_confirm_dead_dispatch_after_finish(ray_cluster: str) -> None:
    executor = RayExecutor()
    await executor.submit(_job("report_env.py", id="x-t-e1-0300"))
    started = Job.Started(kind="dispatch", job_id="x-t-e1-0300", task_name="x-t-e1-0300")
    assert await _eventually(lambda: executor.confirm_dead(started), True, timeout_s=10.0)
    assert await executor.confirm_dead(Job.Started(kind="process", job_id="x", pid=1, pgid=1)) is None

async def test_kill_orphan_bounded_for_unknown_task(ray_cluster: str) -> None:
    start = time.monotonic()
    started = Job.Started(kind="dispatch", job_id="x-none", task_name="x-none")
    assert await RayExecutor(cancel_margin_s=1.0).kill_orphan(started) is False
    assert time.monotonic() - start < 3.0

# tests/e2e/test_ray_cancel.py
pytestmark = pytest.mark.ray

async def test_soft_cancel_gives_partial_output(ray_cluster: str, tmp_path: Path) -> None:
    out, _ = await _cancel_and_capture(RayExecutor(), _partial_job(tmp_path / "ready"), ready=(tmp_path / "ready").exists)
    assert out.status == RunStatus.CANCELLED and out.partial and out.data["step"] >= 1

async def test_force_cancel_kills_grandchildren(ray_cluster: str, tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    job = _tool_job(StallCollectTool, inputs={"pid_file": str(pid_file)}, grace_s=1.0)   # collect() sleeps 60 s
    out, cancel_to_return_s = await _cancel_and_capture(RayExecutor(cancel_margin_s=1.0), job, ready=pid_file.exists)
    assert cancel_to_return_s >= 2.0                                    # grace_s + cancel_margin_s: the force step ran
    await _eventually_async(lambda: worker_dead(job.id) is True, timeout_s=10.0)
    await _eventually_async(lambda: not _alive(int(pid_file.read_text())), timeout_s=5.0)
    assert out.status == RunStatus.CANCELLED and not out.partial        # the ref raised

# tests/_exec/test_executor.py
def test_schedule_timeout_scope() -> None:
    assert current_schedule_timeout() is None
    with schedule_timeout_scope(600.0):
        assert current_schedule_timeout() == 600.0
    assert current_schedule_timeout() is None

# tests/_engine/test_run.py
async def test_every_submit_sees_schedule_timeout(...) -> None:
    # a recording executor stores current_schedule_timeout() at each submit; the plan has one tool run and one
    # ScriptJudge (its ctx.execute job goes through JudgeContext); run.toml sets schedule_timeout_s = 42.0
    assert executor.schedule_timeouts and set(executor.schedule_timeouts) == {42.0}
    assert {job.kind for job in executor.jobs} == {"tool", "command"}

# tests/test_public_api.py
def test_ray_all() -> None:
    pytest.importorskip("swarm_arena.ray")
    assert swarm_arena.ray.__all__ == ["RayExecutor"]

def test_ray_facade_publishes_methods() -> None:     # R0: the facade's publish call, not a hand-set __module__
    pytest.importorskip("swarm_arena.ray")
    cls = swarm_arena.ray.RayExecutor
    assert {cls.__module__, cls.Config.__module__, cls.submit.__module__, cls.confirm_dead.__module__} == {"swarm_arena.ray"}
```

Test helpers in `tests/ray/test_executor.py`: `_SpyRunJob(order)` has `options(**kw) -> _SpyRunJob` (keeps `kw`) and `remote(payload) -> ObjectRef` (appends `"remote"` to `order`, then returns `run_job.options(**kw).remote(payload)` with the real `run_job`). `_sleep_job(seconds, resources)` is a command job that runs `python -c "import time; time.sleep(<seconds>)"`. `_crash_run_job` is a `@ray.remote(max_retries=0)` function that calls `os._exit(1)`. `_tool_job(tool_cls, inputs, grace_s)` is a `kind="tool"` job whose tool `Ref` points at `_fakes.tools:<tool_cls>`. `tests/e2e/test_ray_cancel.py` has its own copies of `_cancel_and_capture`, `_partial_job`, and `_tool_job`, and imports `worker_dead` from `swarm_arena.ray._state`.

The `s3` test: moto runs inside the pytest process and starts after `ray start`, so Ray workers cannot reach it, and the test skips. With `SWARM_ARENA_TEST_S3_ENDPOINT` and the `AWS_*` credentials set before the session (the M2 MinIO path), the `ray_cluster` fixture (Task 5) also passes `AWS_ENDPOINT_URL_S3=$SWARM_ARENA_TEST_S3_ENDPOINT` to `ray start`, so the raylet and its workers use the same S3.

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/ray/test_executor.py tests/e2e/test_ray_cancel.py -m "ray and not s3" -v; uv run pytest tests/_exec/test_executor.py tests/_engine/test_run.py tests/test_public_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'RayExecutor'`, and with `ImportError: cannot import name 'schedule_timeout_scope'`.

- [ ] **Step 3: Implement with the signatures and the pinned behavior above. Write `docs/api/ray.md`, add `ray` to the `docs/api/index.md` toctree, and write the "The Ray executor" section of `docs/running-on-ray.md`.** That section covers the config fields, the resource mapping, the S3 and object-store file paths, the 256 MiB cap, `schedule_timeout_s` (only a request that no node can meet fails; a request that waits for busy nodes keeps its place), the two-step cancel, and GPUs (Ray keeps `CUDA_VISIBLE_DEVICES`; Ray resources are logical, so the rlimits still apply). Add a `CHANGELOG.md` `Unreleased` entry. In the roadmap, change the M4 exit criteria command to `uv run pytest -m "ray and not s3" -v`.

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/ray tests/e2e/test_ray_cancel.py -m "ray and not s3" -v && uv run pytest tests/_exec tests/_engine tests/test_public_api.py tests/test_docstrings.py -v && uv run ty check && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: PASS. Run the `s3` test with `uv run pytest tests/ray/test_executor.py -m "ray and s3" -v`; it skips unless `SWARM_ARENA_TEST_S3_ENDPOINT` was set before the session.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/ray src/swarm_arena/_exec/executor.py src/swarm_arena/_engine tests docs CHANGELOG.md
git commit -m "feat: add RayExecutor" -m "Commits a dispatch intent (job id = task name) before submit, maps Resources to task options, moves files by sha256 through S3 or the object store, sends a FunctionTool's driver-side import root, fails only a run that no node can meet after schedule_timeout_s, and cancels in two steps in a shielded scope, handing the partial output to Job.on_partial as LocalExecutor does. The engine passes RunConfig.schedule_timeout_s to executors through a private context variable, so no public signature changes. Roadmap: the M4 exit command deselects the one ray-and-s3 test."
```

---

### Task 8: Dispatch intent in the core, and Ray runs on resume

**Files:**
- Modify: `src/swarm_arena/_agent/tool_run.py` (`on_started` journals `tool_dispatch` for `kind="dispatch"`)
- Modify: `src/swarm_arena/_journal/replay.py` (`RunState.started` folds `tool_dispatch`)
- Modify: `src/swarm_arena/_engine/resume.py` (a dispatch record's `run_lost` gets `kind`, `task_name`, and `confirmed_dead` from M1b's one `kill_orphan` call)
- Modify: roadmap R3.1 (`_journal/replay.py` `+M4`)
- Test: `tests/_agent/test_tool_run.py`, `tests/_journal/test_replay.py`, `tests/_engine/test_resume.py`, `tests/e2e/test_ray_dispatch_crash.py` (marker `ray`)

**Interfaces:**
- Consumes: `JournalWriter.append_durable` (R4.13.3), `Job.Started`, `Executor.kill_orphan` (R4.7), M1b's round admission check in `on_started` and `LimitReached("stop")` (R4.13.10), `RunState.started_commands` and the `command_start` event (R4.13.4, R4.14), `RayExecutor` (Task 7), `find_task` (Task 6), `Swarm.resume` (R4.1).
- Produces: event `tool_dispatch` (write-through, R4.14). Its payload is the M1b `tool_start` payload shape: `{"candidate_id": str, "started": <Job.Started JSON with kind="dispatch">, "reservation_id": str}`. `RunState.started[job_id]` holds the `Job.Started` until a `tool_end` for that job. The `run_lost` payload for a dispatch record gains `{"kind": "dispatch", "task_name": str, "confirmed_dead": bool}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_agent/test_tool_run.py
async def test_dispatch_start_is_durable_before_submit(...) -> None:
    # a fake executor awaits on_started(kind="dispatch"), then, before it returns, records the committed event types
    # in fake.committed_at_submit and journal.head in fake.head_at_submit
    assert fake.committed_at_submit[-1] == "tool_dispatch"
    assert fake.head_at_submit.seq == _event(journal, "tool_dispatch").seq     # the commit covered the intent

async def test_closed_admission_refuses_dispatch_before_intent(rt_env: RtEnv, ...) -> None:
    # the same fake executor, with close_first=True: submit calls rt_env.inflight.close() (after Inflight.add), then
    # awaits on_started(kind="dispatch"); it records committed_at_submit only after on_started returns
    with pytest.raises(LimitReached) as err:
        await execute_tool_run(rt, {"marker": str(rt_env.marker)}, parents=(), replicate=None, edits=None)
    assert err.value.limit == "stop" and fake.committed_at_submit is None    # on_started raised: nothing submitted
    res = rt_env.events("reserve")[-1].payload["reservation"]["id"]
    assert "tool_dispatch" not in rt_env.committed_types() and rt_env.charged(res) == Amount()

# tests/_journal/test_replay.py
DISPATCH = {"candidate_id": "c-a-e1-0001", "reservation_id": "r-a-e1-0001",
            "started": {"kind": "dispatch", "job_id": "c-a-e1-0001", "task_name": "c-a-e1-0001"}}

def test_replay_dispatch_without_end_is_started() -> None:
    state = RunState.from_events(_events(("reserve", ...), ("tool_dispatch", DISPATCH)))
    assert state.started["c-a-e1-0001"].kind == "dispatch"

def test_replay_dispatch_with_end_is_not_started() -> None:
    assert "c-a-e1-0001" not in RunState.from_events(_events(("tool_dispatch", DISPATCH), ("tool_end", ...))).started

# tests/_engine/test_resume.py
async def test_resume_marks_dispatch_lost_after_kill_orphan(...) -> None:   # fake executor: kill_orphan → True;
    lost = _events(journal, "run_lost")[0].payload                          # the crashed journal has tool_dispatch with
    assert lost["kind"] == "dispatch" and lost["task_name"] == "c-a-e1-0001" and lost["confirmed_dead"] is True   # DISPATCH
    assert state.candidates["c-a-e1-0001"].output.status == RunStatus.LOST     # reservation stays charged (M1b)

async def test_resume_kills_dispatched_command_orphan(...) -> None:   # crashed journal: command_start with started
    # {"kind": "dispatch", "job_id": "x-j-e1-0001", "task_name": "x-j-e1-0001"} and no command_end; spy executor
    started = Job.Started(kind="dispatch", job_id="x-j-e1-0001", task_name="x-j-e1-0001")
    assert spy.kill_orphan_calls.count(started) == 1                  # M1b's command-orphan loop; no second call

async def test_resume_counts_dispatch_toward_tool_runs(...) -> None:   # max_tool_runs_per_round = 1; the crashed
    # journal's unfinished round has tool_dispatch and tool_end for agent a (no tool_start)
    new = _events_since_takeover(journal, {"tool_start", "tool_dispatch"})
    assert not [e for e in new if _agent_of(e) == "a"]                 # _agent_of: the agent of the event's candidate

@pytest.mark.ray
async def test_resume_dispatch_never_submitted_is_lost(ray_cluster: str, tmp_path: Path) -> None:  # Review Focus 5
    pytest.importorskip("swarm_arena.ray")                                  # the core CI job collects this module
    from swarm_arena.ray import RayExecutor
    run_id = _crashed_run_with_dispatch(tmp_path, job_id="c-a-e1-0009")     # committed tool_dispatch (DISPATCH shape),
    start = time.monotonic()                                                # no Ray task
    await Swarm(_config(tmp_path, executor=RayExecutor(cancel_margin_s=1.0), lease_ttl_s=1.0)).resume(run_id, force=True)
    assert time.monotonic() - start < 30.0
    assert _run_lost(tmp_path, run_id, "c-a-e1-0009")["confirmed_dead"] is False

# tests/e2e/test_ray_dispatch_crash.py
pytestmark = pytest.mark.ray

def test_driver_killed_after_dispatch_then_resume(ray_cluster: str, tmp_path: Path) -> None:
    driver = subprocess.Popen([SWARM_ARENA, "run", "--yes", str(run_toml), str(plan_toml)], env=_env(ray_cluster))
    job_id = _wait_for_committed(tmp_path, "tool_dispatch", timeout_s=60.0)["started"]["job_id"]
    _wait_until(lambda: (i := _ray_find_task(ray_cluster, job_id)) is not None and i.state not in PENDING_STATES,
                timeout_s=15.0)                                             # .remote() ran and the task started
    driver.send_signal(signal.SIGKILL); driver.wait()
    run_dir = _run_dir(tmp_path); run_id = run_dir.name                    # RUN_DIR = <run_root>/<run-id> (M1b Task 28)
    assert subprocess.run([SWARM_ARENA, "resume", str(run_dir), "--force"], env=_env(ray_cluster)).returncode == 0
    assert _run_lost(tmp_path, run_id, job_id)["confirmed_dead"] is True
    assert _ray_find_task(ray_cluster, job_id) is not None                 # resume found it by name
    assert _solution(tmp_path, run_id)["schema_version"] == 1             # the resumed run finished
```

`_run_dir(tmp_path) -> Path` returns the only directory under the `run.toml`'s `run_root`; `resume` takes that RUN_DIR, not a run id (R4.13.18). The e2e plan uses one `ScriptTool` agent running `tests/_fakes/tools.py`'s `ignore_sigterm.py` with `grace_s = 1.0`, `RayExecutor` in `run.toml`, `lease_ttl_s = 2`, `limits.tool_timeout_s = 20`, and 1 round. After resume, `ScriptedManager` sends the lost request again; that second run ends by its 20 s tool timeout (plus 1 s grace), so the resumed run finishes in bounded time. The SIGKILL comes less than 15 s after the dispatch commit, so the first run is still inside its 20 s tool timeout.

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/_agent/test_tool_run.py tests/_journal/test_replay.py tests/_engine/test_resume.py -v; uv run pytest tests/_engine/test_resume.py tests/e2e/test_ray_dispatch_crash.py -m "ray and not s3" -v`
Expected: FAIL. The new assertions fail, for example `KeyError: 'c-a-e1-0001'` and `assert 'tool_start' == 'tool_dispatch'`. `test_resume_kills_dispatched_command_orphan` can pass already, because M1b's command-orphan loop covers dispatch records; it guards against a second call. `test_closed_admission_refuses_dispatch_before_intent` can pass already, because M1b's admission check in `on_started` covers every kind; it guards the order (the check stays before `tool_dispatch`).

- [ ] **Step 3: Implement**

- `on_started`: `kind="dispatch"` → `append_durable(EventDraft(type="tool_dispatch", payload={"candidate_id", "started", "reservation_id"}))`, the same keys as `tool_start`. `kind="process"` stays `tool_start`. M1b's admission check stays first in the callback (R4.13.10: refused at `Inflight.add` and again in `on_started`): with a closed admission it raises `LimitReached("stop")` before `tool_dispatch`, the refused run settles `Amount()`, and `RayExecutor` submits nothing (its step 3 already propagates an `on_started` exception).
- Replay: fold `tool_dispatch` into `started`, exactly like `tool_start` (`payload["started"]` is the `Job.Started`).
- Resume: M1b Task 27 already calls `await executor.kill_orphan(started)` once for every `state.started` record and every `state.started_commands` entry, of any kind. The host and start-time checks are inside the executor's `kill_orphan` (M1a `kill_orphan` for `kind="process"`), and `RayExecutor.kill_orphan` handles `kind="dispatch"` (Task 7). So M4 adds no second `kill_orphan` call. For a `state.started` record with `kind="dispatch"`, keep the bool that M1b's call returns as `confirmed`, and journal `run_lost` with the payload above (`confirmed_dead=confirmed`) and the M1b lost-run fields. Dispatched command orphans need no code change: M1b's command-orphan loop already covers them (M1b's `command_start` callback already gets `Started(kind="dispatch")` from `RayExecutor`). M4 adds only `test_resume_kills_dispatched_command_orphan`, which checks for exactly one call.
- Kept counters: the `tool_runs` counter of the unfinished round (M1b Task 27 counts `tool_start` and `run_lost`) also counts `tool_dispatch`, the same way it counts `tool_start`.

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/_agent tests/_journal tests/_engine -v && uv run pytest tests/_engine/test_resume.py tests/e2e/test_ray_dispatch_crash.py -m "ray and not s3" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_agent/tool_run.py src/swarm_arena/_journal/replay.py src/swarm_arena/_engine/resume.py tests docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat: commit Ray dispatch intents and mark dead-driver Ray runs lost on resume" -m "A Ray tool run's tool_dispatch event is written through before the task is submitted. Resume finds each in-flight task by its job id (the task name), waits for Ray to confirm its worker is dead, and journals run_lost with confirmed_dead. Dispatched command jobs get the same kill_orphan wait, and a dispatch counts toward the round's tool runs like a start."
```

---

### Task 9: `swarm-arena submit`

**Files:**
- Create: `src/swarm_arena/ray/_submit.py`, `src/swarm_arena/_cli/cluster_cmds.py`
- Modify: `src/swarm_arena/_cli/main.py` (call `cluster_cmds.register`), roadmap R3.1 (`_cli/main.py` `+M4`) and R4.13.16 (`submit_driver(address: str | None, ...)`)
- Modify: `docs/running-on-ray.md` (section "Running the driver as a Ray Job")
- Test: `tests/ray/test_submit.py` (default markers; `collect_ignore` without `ray`), `tests/_cli/test_cluster_cmds.py`

**Interfaces:**
- Consumes: `main` and `register` conventions (R4.13.18).
- Produces:

```python
# ray/_submit.py
def driver_entrypoint(files: Sequence[str]) -> str: ...      # shlex.join(["swarm-arena", "run", "--yes", *files])
def submit_runtime_env(*, working_dir: str | None, pip: Sequence[str]) -> dict[str, Any] | None: ...
    # only "working_dir" and "pip" keys; never "env_vars"; None when both are empty
def submit_driver(address: str | None, entrypoint: str, *, runtime_env: Mapping[str, Any] | None) -> str: ...
    # JobSubmissionClient(address).submit_job(entrypoint=entrypoint, runtime_env=runtime_env) → submission id;
    # address None → Ray reads RAY_ADDRESS. _submit.py has `from ray.job_submission import JobSubmissionClient` at
    # module level and calls it through that module global, so a test patches swarm_arena.ray._submit.JobSubmissionClient
# _cli/cluster_cmds.py
def register(sub: argparse._SubParsersAction) -> None: ...  # "submit" here; "reconcile" is added in Task 16
```

CLI: `swarm-arena submit [--address URL] [--working-dir DIR] [--pip SPEC]... [--yes] RUN_TOML PLAN...`. Without `--yes`, it exits with code 2, submits nothing, and prints `submit starts the driver unattended: check the estimate with "swarm-arena plan", then pass --yes`. On success, it prints the submission id on stdout and exits 0. A Ray error exits 1. The handler imports `swarm_arena.ray._submit` inside its body.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ray/test_submit.py
def test_driver_entrypoint_quotes_paths() -> None:
    assert driver_entrypoint(["run.toml", "plans dir/a.toml"]) == "swarm-arena run --yes run.toml 'plans dir/a.toml'"

def test_submit_runtime_env_never_has_env_vars() -> None:
    assert submit_runtime_env(working_dir=".", pip=["deepretro-swarm"]) == {"working_dir": ".", "pip": ["deepretro-swarm"]}
    assert submit_runtime_env(working_dir=None, pip=[]) is None

def test_submit_driver_calls_job_submission_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(returns="raysubmit_123")
    monkeypatch.setattr("swarm_arena.ray._submit.JobSubmissionClient", fake)
    assert submit_driver("http://head:8265", "swarm-arena run --yes r.toml p.toml", runtime_env=None) == "raysubmit_123"
    assert fake.address == "http://head:8265" and fake.kwargs == {"entrypoint": "swarm-arena run --yes r.toml p.toml",
                                                                  "runtime_env": None}

# tests/_cli/test_cluster_cmds.py
def test_submit_without_yes_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["submit", "run.toml", "plan.toml"]) == 2 and "pass --yes" in capsys.readouterr().err

def test_submit_prints_submission_id(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("swarm_arena.ray")
    monkeypatch.setattr("swarm_arena.ray._submit.submit_driver", lambda *a, **k: "raysubmit_9")
    assert main(["submit", "--yes", "run.toml", "plan.toml"]) == 0 and capsys.readouterr().out.strip() == "raysubmit_9"

def test_help_loads_no_ray() -> None:
    code = "import sys\nfrom swarm_arena._cli.main import main\ntry: main(['submit', '--help'])\nexcept SystemExit: pass\n" \
           "assert 'ray' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
```

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/ray/test_submit.py tests/_cli/test_cluster_cmds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena.ray._submit'` and an argparse `invalid choice: 'submit'`.

- [ ] **Step 3: Implement. Write the docs section.** It covers: the driver runs inside the cluster as a Ray Job with `ray.init(address="auto")`; `swarm-arena submit --address http://<head>:8265 --working-dir . --yes run.toml plan.toml`; Ray Client is not used; secrets never go in `runtime_env`; the worker image or `--pip` must hold `swarm-arena[ray]` and the connector. The handler calls `_submit.submit_driver` through the module (not a `from` import), so the test can patch it.

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/ray/test_submit.py tests/_cli -v && uv run ruff check . && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/ray/_submit.py src/swarm_arena/_cli tests/ray/test_submit.py tests/_cli/test_cluster_cmds.py docs docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat: add swarm-arena submit to run the driver as a Ray Job" -m "Wraps JobSubmissionClient.submit_job with 'swarm-arena run --yes'. The runtime_env only ever holds working_dir and pip, so no secret reaches Ray's job metadata. Requires --yes, because the job runs unattended. Roadmap R4.13.16: submit_driver takes address None, so Ray reads RAY_ADDRESS when --address is not given."
```

---
### Task 10: Ray end to end: the conformance kit and Rosenbrock on Ray

**Files:**
- Create: `tests/e2e/test_rosenbrock_ray.py` (marker `ray`)
- Modify: `docs/writing-a-connector.md` (section "Connector CI"), `docs/running-on-ray.md` (complete the page), `README.md` (section "Run on a Ray cluster"), `CHANGELOG.md`
- Test: `tests/testing/test_conformance.py`

**Interfaces:**
- Consumes: `check_tool` (R4.12), `RayExecutor` (Task 7), `Swarm.run`, `RunConfig.load`, `examples/rosenbrock/` (M1b).
- Produces: proof that `check_tool(tool, inputs, *, executor=RayExecutor(...))` and the Rosenbrock example work on Ray. `check_tool` needs no change: M1a already closes only an executor that it creates, and `RayExecutor` sends a `FunctionTool`'s driver-side import root to the worker (Task 7). The temporary `artifact_root` of `check_tool` is local, so a Ray run moves its files through the object store.

- [ ] **Step 1: Write the tests**

```python
# tests/testing/test_conformance.py
@pytest.mark.ray
async def test_check_tool_on_ray_matches_local(ray_cluster: str, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("swarm_arena.ray")                    # the core CI job collects this module
    from swarm_arena.ray import RayExecutor
    monkeypatch.syspath_prepend(str(EXAMPLES / "rosenbrock"))   # only the driver's sys.path has rosen
    tool = FunctionTool("rosen:gradient_descent")
    inputs = {"lr": 0.001, "steps": 200, "x0": -1.0, "y0": 1.0}
    remote = await check_tool(tool, inputs, executor=RayExecutor())
    assert remote.status == RunStatus.OK and remote.data == (await check_tool(tool, inputs)).data

# tests/e2e/test_rosenbrock_ray.py
pytestmark = pytest.mark.ray

async def test_rosenbrock_on_ray(ray_cluster: str, tmp_path: Path) -> None:
    config = _rosen_config(tmp_path).model_copy(update={"executor": Ref(path="swarm_arena.ray:RayExecutor")})
    finalists = await Swarm(config).run(Plan.load(EXAMPLES / "rosenbrock" / "plan.toml"))
    solution = json.loads(_run_dir(tmp_path).joinpath("goals", "rosenbrock", "solution.json").read_text())
    assert len(finalists["rosenbrock"]) == 2 and len(solution["finalists"]) == 2
    types = _committed_types(tmp_path)
    assert "tool_dispatch" in types and "tool_start" not in types
    assert all((_run_dir(tmp_path) / "goals" / "rosenbrock" / "logs" / f"{c.id}.out").exists()
               for c in _candidates(tmp_path))
```

Use the M1b `test_rosenbrock.py` names for the solution fields and the goal id. Do not invent new ones.

- [ ] **Step 2: Run the tests, and fix what they find**

Run: `uv run pytest tests/testing/test_conformance.py tests/e2e/test_rosenbrock_ray.py -m "ray and not s3" -v`
Expected: these integration tests can pass at once, because Tasks 5 and 7 carry the import root and the files. A failure is an integration bug, for example a worker that cannot import `rosen`. Fix it in the owning module, and add a unit test there.

- [ ] **Step 3: Write the docs**

- `docs/writing-a-connector.md` "Connector CI": a GitHub Actions job that runs `uv sync --extra ray`, then `uv run ray start --head --num-cpus=2 --disable-usage-stats`, then `uv run pytest`. Its tests call `check_tool(MyTool(...), inputs, executor=RayExecutor())`, `check_check`, `check_judge`, and `check_manager` with `FakeModel`, so no API key is needed.
- `docs/running-on-ray.md`, completed:
  - token auth: on by default for local clusters in 2.59; remote drivers set `RAY_AUTH_TOKEN` and `RAY_AUTH_MODE=token`;
  - images: the worker environment needs `swarm-arena[ray]`, the connector, and the tool's own venv;
  - the tool venv runs as a subprocess, and `py_executable` is not used;
  - a `FunctionTool` module next to the plan file (or on the driver's `sys.path`) reaches the worker as an import root path, so that path must exist on every node (one host, a shared file system, or the same path in the image); otherwise install the module in the worker image;
  - VMs join as worker nodes with `ray start --address=<head>:6379`;
  - secrets: never in `runtime_env`; set on each node, or in `secrets.file` on each node; `RayExecutor(secret_resource="llm_keys")` and `ray start --resources='{"llm_keys": 1}'` on the nodes that have keys;
  - how resume finds a dead driver's tasks.
- `README.md`: 6 lines with `executor = { path = "swarm_arena.ray:RayExecutor" }` in `run.toml`, `pip install 'swarm-arena[ray]'`, and `swarm-arena submit --yes …`.

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/testing -v && uv run pytest -m "ray and not s3" -v && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/testing/test_conformance.py tests/e2e/test_rosenbrock_ray.py docs README.md CHANGELOG.md
git commit -m "test: run the conformance kit and the Rosenbrock example on Ray" -m "check_tool with a RayExecutor and the Rosenbrock plan both run a FunctionTool that only the driver's sys.path imports. Adds the Running on Ray page and connector CI instructions."
```

---

### Task 11: Gateway adapter protocol and the strict-profile check

**Files:**
- Create: `src/swarm_arena/_gateway/__init__.py` (empty), `src/swarm_arena/_gateway/adapter.py`, `src/swarm_arena/_gateway/profile.py`
- Create: `tests/_fakes/gateway.py` (only `strict_settings()`, `strict_models()`, and `ABSENT_FAILURES` in this task)
- Test: `tests/_gateway/test_adapter.py`, `tests/_gateway/test_profile.py`

**Interfaces:**
- Consumes: `FrozenModel`, `Usage`; the gateway names from Task 1 items 1 to 6, 11, 12, 22, and 23.
- Produces (R4.13.15, plus the private names below):

```python
# _gateway/adapter.py
class GatewayError(RuntimeError): ...       # [plan] message: method, path, and HTTP status or error type (Task 12);
                                            # never a key value
class ProfileReport(FrozenModel):
    verified: bool; failures: list[str]; batch_write_interval_s: float | None; request_timeout_s: float | None
class VirtualKey(FrozenModel):
    key_id: str; alias: str
class SpendRow(FrozenModel):
    litellm_call_id: str; model: str; usage: Usage; cost_usd: float; ts: datetime; metadata: dict[str, Any]
class GatewayAdapter(Protocol): ...          # exactly as R4.13.15

# _gateway/profile.py — names from Task 1; None when Task 1 recorded the item as `absent`
BUDGET_RESERVATIONS: Final[str | None]; FAIL_CLOSED: Final[str | None]; MAX_REQUEST_DURATION: Final[str | None]
BATCH_WRITE_INTERVAL: Final[str | None]; VIRTUAL_KEYS_REQUIRED: Final[str | None]; PASS_THROUGH: Final[str | None]
MAX_TOKENS_BOUND: Final[str | None]          # Task 1 item 22
RESERVED_ROUTES_ONLY: Final[str | None]      # Task 1 item 23
def check_profile(settings: Mapping[str, Any], models: Sequence[Mapping[str, Any]], *,
                  request_timeout_s: float) -> ProfileReport: ...
    # settings: general_settings and litellm_settings of the config read endpoint, flattened into one mapping;
    # models: the /model/info "data" list. Pure; verified == (failures == []).
```

An absent setting (its constant is `None`) always fails its rule: `check_profile` lists that rule's failure text whatever the settings hold. For the duration rule this is `"no maximum request duration is enforced"`, with `request_timeout_s=None` in the report. `BATCH_WRITE_INTERVAL` `None` gives `batch_write_interval_s=None`, which is not a failure (see below).

Failure texts (exact, in this order):
- `"budget reservations are disabled"`
- `"spend enforcement does not fail closed when spend counters are stale"`
- `f"model {name}: no price (input and output cost per token)"`
- `f"model {name}: no max output token bound"`
- `"request max_tokens is not bounded per key"`
- `"routes without budget reservations are enabled"`
- `"no maximum request duration is enforced"`
- `f"maximum request duration {d:g} s is longer than llm_gateway.request_timeout_s {t:g} s"`
- `"virtual keys are not required"`
- `"pass-through endpoints are configured"`

The per-model `"no max output token bound"` rule stays, but it is never enough alone: an advertised model maximum does not bound a request's `max_tokens` per key. An unknown batch write interval gives `batch_write_interval_s=None`, and that is not a failure (Task 16 then labels spend `estimated`).

- [ ] **Step 1: Write the failing tests**

```python
def test_strict_profile_verified() -> None:
    report = check_profile(strict_settings(request_timeout_s=300, batch_write_interval_s=10), strict_models(),
                           request_timeout_s=600.0)
    if ABSENT_FAILURES:                                  # some Task 1 item is `absent`: its rule always fails
        assert report.verified is False and report.failures == ABSENT_FAILURES
    else:
        assert report == ProfileReport(verified=True, failures=[], batch_write_interval_s=10.0, request_timeout_s=300.0)

RULE_ROWS = [                                            # (constant, change, failure); rows of absent rules are dropped
    (BUDGET_RESERVATIONS, {BUDGET_RESERVATIONS: "off"}, "budget reservations are disabled"),
    (FAIL_CLOSED, {FAIL_CLOSED: "off"}, "spend enforcement does not fail closed when spend counters are stale"),
    (MAX_TOKENS_BOUND, {MAX_TOKENS_BOUND: "off"}, "request max_tokens is not bounded per key"),
    (RESERVED_ROUTES_ONLY, {RESERVED_ROUTES_ONLY: "off"}, "routes without budget reservations are enabled"),
    (MAX_REQUEST_DURATION, {MAX_REQUEST_DURATION: None}, "no maximum request duration is enforced"),
    (MAX_REQUEST_DURATION, {MAX_REQUEST_DURATION: 900},
     "maximum request duration 900 s is longer than llm_gateway.request_timeout_s 600 s"),
    (VIRTUAL_KEYS_REQUIRED, {VIRTUAL_KEYS_REQUIRED: "off"}, "virtual keys are not required"),
    (PASS_THROUGH, {PASS_THROUGH: "on"}, "pass-through endpoints are configured")]

@pytest.mark.parametrize(("change", "failure"), [(c, f) for name, c, f in RULE_ROWS if name is not None])
def test_each_profile_rule(change: dict[str, Any], failure: str) -> None:
    # strict_models() gives every model an advertised max output, so the MAX_TOKENS_BOUND row also proves that an
    # advertised model maximum does not satisfy the per-key rule
    report = check_profile(strict_settings(**_as_kwargs(change)), strict_models(), request_timeout_s=600.0)
    assert not report.verified and report.failures == _pinned_merge(ABSENT_FAILURES, failure)

@pytest.mark.parametrize("failure", ABSENT_FAILURES or [None])     # one per None constant
def test_absent_setting_always_fails(failure: str | None) -> None:
    if failure is None:
        pytest.skip("no Task 1 item is absent")
    assert failure in check_profile(strict_settings(), strict_models(), request_timeout_s=600.0).failures

def test_unpriced_and_unbounded_models_fail() -> None:
    models = [_model("cheap", price=False), _model("long", max_output=None)]
    assert check_profile(strict_settings(), models, request_timeout_s=600.0).failures == _pinned_merge(
        ABSENT_FAILURES, "model cheap: no price (input and output cost per token)", "model long: no max output token bound")

def test_unknown_batch_interval_is_none_not_failure() -> None:
    report = check_profile(strict_settings(batch_write_interval_s=None), strict_models(), request_timeout_s=600.0)
    assert report.failures == ABSENT_FAILURES and report.batch_write_interval_s is None

def test_virtual_key_holds_no_secret() -> None:
    assert set(VirtualKey.model_fields) == {"key_id", "alias"}
```

`strict_settings(**overrides)` builds the flattened settings with the Task 1 names, and leaves out a name whose constant is `None`. `"off"`, `"on"`, and `None` in `_as_kwargs` map to the real values that Task 1 recorded; `_as_kwargs` has no mapping for an absent name. `tests/_fakes/gateway.py` also defines `ABSENT_FAILURES: list[str]`, the failure texts of the rules whose constant is `None`, in the pinned order (Tasks 12, 14, and 17 import it). `_pinned_merge(absent, *texts)` in the test file returns `absent` plus `texts`, sorted by the pinned order of the failure texts.

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/_gateway -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._gateway'`.

- [ ] **Step 3: Implement both modules with the signatures above**

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/_gateway -v && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_gateway tests/_gateway tests/_fakes/gateway.py
git commit -m "feat: add the gateway adapter protocol and the strict-profile check" -m "check_profile is pure: it reads the gateway's settings and model info and lists every rule of the strict profile (spec 7.3 item 5) that fails, so preflight can decide between a hard and a soft cap. Request max_tokens must be bounded per key and only reserved routes enabled; a setting that LiteLLM 1.105.x does not have always fails its rule."
```

---

### Task 12: LiteLLM Proxy adapter and the fake gateway

**Files:**
- Create: `src/swarm_arena/_gateway/litellm_proxy.py`
- Modify: `tests/_fakes/gateway.py` (`FakeGateway`), `tests/conftest.py` (fixture `fake_gateway`)
- Test: `tests/_gateway/test_litellm_proxy.py` (and one `gateway_live` test)

**Interfaces:**
- Consumes: `GatewayAdapter`, `VirtualKey`, `SpendRow`, `ProfileReport`, `GatewayError`, `check_profile` (Task 11); `GatewayConfig` (R4.8); `ConfigError` (R4.13.1); `utc_now` (R4.13.0); the endpoint facts from Task 1 items 7 to 13 and 22.
- Produces:

```python
# _gateway/litellm_proxy.py
class LiteLLMProxy:                                    # satisfies GatewayAdapter
    terminal_accounting: bool = False
    def __init__(self, *, url: str, master_key: str, request_timeout_s: float, http_timeout_s: float = 30.0) -> None: ...
    async def preflight(self) -> ProfileReport: ...    # health, config read, /model/info → check_profile
    async def create_key(self, *, alias: str, max_budget_usd: float, duration_s: float, metadata: Mapping[str, Any],
                         default_estimated_output_tokens: int) -> tuple[VirtualKey, str]: ...
        # [plan] when Task 1 item 22 names a per-key max_tokens field, the body also sets it to
        # default_estimated_output_tokens
    async def block_key(self, key: VirtualKey) -> None: ...
    async def closed_at(self, key: VirtualKey) -> datetime | None: ...   # blocked or 404 → utc_now(); else None
    async def spend_rows(self, key: VirtualKey) -> list[SpendRow]: ...   # every page
    async def delete_key(self, key: VirtualKey) -> None: ...
def adapter_for(config: GatewayConfig, environ: Mapping[str, str]) -> LiteLLMProxy: ...
    # master key = environ[config.master_key_env]; missing → ConfigError(f"set {config.master_key_env} for the LLM gateway")

# tests/_fakes/gateway.py
class FakeGateway:                                     # stdlib ThreadingHTTPServer on 127.0.0.1:0, context manager
    def __init__(self, *, strict: bool = True, batch_write_interval_s: float | None = 0.0,
                 max_request_duration_s: float | None = 0.1) -> None: ...
    # max_request_duration_s is the enforced request duration that the config read endpoint reports; 0.1 s passes the
    # strict-profile rule for any request_timeout_s >= 0.1 (Task 17 reconciles with request_timeout_s = 0.2)
    url: str
    requests: list[tuple[str, str, dict[str, Any]]]    # (method, path, JSON body or query)
    keys: dict[str, dict[str, Any]]                    # key_id → {"alias", "secret", "blocked", "deleted", "max_budget", ...}
    fail: dict[str, int]                               # path → HTTP status to return (fault injection)
    delay: dict[str, float]                            # path → seconds to wait before the reply
    malformed: set[str]                                # paths that reply HTTP 200 with the body b"{not json"
    chat_visible_after_s: float = 0.0                  # a chat spend row shows in /spend/logs/v2 only after this delay
    def add_spend(self, key_id: str, *, model: str, cost_usd: float, tokens: tuple[int, int], call_id: str,
                  visible_after_s: float = 0.0) -> None: ...
    # also serves POST /v1/chat/completions: a Bearer virtual key that is not blocked adds one spend row
    # (model from the body, cost 0.01, a new call id) and returns a minimal completion that carries that spend row's
    # call id twice: the `x-litellm-call-id` response header and the response body `id` (so a tool can report it in
    # swarm/llm_calls.jsonl as litellm_call_id); a blocked key → 401
```

All HTTP goes through `urllib.request` inside `anyio.to_thread.run_sync`, with `Authorization: Bearer <master key>`, a JSON body, and `timeout=http_timeout_s`. Every adapter method parses the response body with `json.loads` and a private pydantic model. A non-2xx status maps to `GatewayError(f"{method} {path}: HTTP {status}")`, except a 404 from `/key/info`. Every transport or parse failure maps to `GatewayError(f"{method} {path}: {type(err).__name__}")`, raised `from None`: `OSError` (this covers `URLError`, `ConnectionError`, and `TimeoutError`), `http.client.HTTPException`, and `ValueError` (this covers `JSONDecodeError` and `ValidationError`). The path has no query string. Cancellation is never caught, and no message ever contains a key value.

- [ ] **Step 1: Write the failing tests**

```python
async def test_preflight_strict_verified(fake_gateway: FakeGateway) -> None:
    report = await _proxy(fake_gateway).preflight()
    assert (report.verified, report.failures) == (not ABSENT_FAILURES, ABSENT_FAILURES)   # Task 11: absent rules fail

async def test_preflight_not_strict_lists_failures() -> None:
    with FakeGateway(strict=False) as gw:
        assert (await _proxy(gw).preflight()).failures != []

async def test_create_key_sends_alias_budget_duration_metadata(fake_gateway: FakeGateway) -> None:
    key, secret = await _proxy(fake_gateway).create_key(alias="cand-c-3f9a1c2b5d6e7f80-e1-0193", max_budget_usd=1.0,
                                                        duration_s=86_400.0, metadata={"candidate_id": "c-3f9a1c2b5d6e7f80-e1-0193"},
                                                        default_estimated_output_tokens=4096)
    body = _last(fake_gateway, "/key/generate")
    assert body["key_alias"] == "cand-c-3f9a1c2b5d6e7f80-e1-0193" and body["max_budget"] == 1.0 and body["duration"] == "86400s"
    assert key.alias == "cand-c-3f9a1c2b5d6e7f80-e1-0193" and secret.startswith("sk-") and secret not in key.model_dump_json()
    # default_estimated_output_tokens == 4096 is asserted at the location Task 1 item 7 recorded; when item 22 names a
    # per-key max_tokens field, that field == 4096 is asserted at the location item 22 recorded

async def test_block_then_closed_at(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    assert await proxy.closed_at(key) is None
    await proxy.block_key(key)
    assert (await proxy.closed_at(key)) is not None

async def test_closed_at_deleted_key(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    await proxy.delete_key(key)
    assert (await proxy.closed_at(key)) is not None

async def test_spend_rows_all_pages(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    for i in range(130):                                            # more than one page of the fake
        fake_gateway.add_spend(key.key_id, model="openai/gpt-6", cost_usd=0.01, tokens=(10, 2), call_id=f"id-{i}")
    rows = await proxy.spend_rows(key)
    assert len(rows) == 130 and rows[0].usage.input_tokens == 10 and rows[0].cost_usd == 0.01

async def test_errors_never_contain_secrets(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    fake_gateway.fail["/spend/logs/v2"] = 500
    with pytest.raises(GatewayError) as err:
        await proxy.spend_rows(key)
    assert "HTTP 500" in str(err.value) and MASTER not in str(err.value) and _secret(fake_gateway) not in str(err.value)

PATHS = {"create_key": "/key/generate", "block_key": "/key/block", "spend_rows": "/spend/logs/v2"}

@pytest.mark.parametrize("call", list(PATHS))
@pytest.mark.parametrize("fault", ["refused", "slow", "malformed"])
async def test_transport_faults_raise_gateway_error(fake_gateway: FakeGateway, fault: str, call: str) -> None:
    proxy, key = await _proxy_and_key(fake_gateway, http_timeout_s=0.2)
    path = PATHS[call]
    if fault == "refused":
        proxy = _proxy_at(f"http://127.0.0.1:{_closed_port()}", http_timeout_s=0.2)   # nothing listens there
    elif fault == "slow":
        fake_gateway.delay[path] = 1.0
    else:
        fake_gateway.malformed.add(path)
    with pytest.raises(GatewayError) as err:
        await _call(proxy, call, key)                    # create_key with the fixed test arguments, or (key)
    assert path in str(err.value) and MASTER not in str(err.value) and _secret(fake_gateway) not in str(err.value)

async def test_fake_chat_returns_spend_call_id(fake_gateway: FakeGateway) -> None:
    proxy, key = await _proxy_and_key(fake_gateway)
    headers, body = _chat(fake_gateway, _secret(fake_gateway), model="openai/gpt-6")   # urllib POST /v1/chat/completions
    rows = await proxy.spend_rows(key)
    assert headers["x-litellm-call-id"] == body["id"] == rows[-1].litellm_call_id

def test_adapter_for_missing_master_key_env() -> None:
    with pytest.raises(ValueError, match="set LITELLM_MASTER_KEY for the LLM gateway"):
        adapter_for(GatewayConfig(url="http://gw:4000"), {})

@pytest.mark.gateway_live
async def test_live_gateway_profile_and_key_cycle() -> None:   # SWARM_ARENA_LIVE_GATEWAY_URL and LITELLM_MASTER_KEY
    proxy = adapter_for(GatewayConfig(url=os.environ["SWARM_ARENA_LIVE_GATEWAY_URL"]), os.environ)
    assert isinstance((await proxy.preflight()).verified, bool)
    key, _ = await proxy.create_key(alias=f"cand-live-{uuid4().hex[:8]}", max_budget_usd=0.01, duration_s=600.0,
                                    metadata={}, default_estimated_output_tokens=256)
    await proxy.block_key(key)
    assert await proxy.closed_at(key) is not None
    await proxy.delete_key(key)
```

`_closed_port()` binds a socket to `127.0.0.1:0`, reads its port, and closes it. `_proxy_at(url, http_timeout_s)` builds a `LiteLLMProxy` with the test `MASTER` key. `_proxy_and_key(gw, http_timeout_s=30.0)` also creates one key.

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/_gateway/test_litellm_proxy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._gateway.litellm_proxy'`.

- [ ] **Step 3: Implement `LiteLLMProxy`, `adapter_for`, and `FakeGateway` with the signatures and the Task 1 endpoint names**

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/_gateway -v && uv run ruff check . && uv run ty check`
Expected: PASS. The `gateway_live` test is not collected by default.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_gateway/litellm_proxy.py tests/_fakes/gateway.py tests/conftest.py tests/_gateway/test_litellm_proxy.py
git commit -m "feat: add the stdlib LiteLLM Proxy adapter and a fake gateway for tests" -m "The adapter uses urllib only: preflight of the strict profile, key create, block, info, delete, and spend logs v2, with terminal_accounting False. HTTP, transport, and parse failures all become GatewayError, which names the method and path (and the status or error type), never a key."
```

---

### Task 13: One key per tool run (`_gateway/keys.py`)

**Files:**
- Create: `src/swarm_arena/_gateway/keys.py`
- Modify: `src/swarm_arena/_exec/executor.py` (`Job.drop_env`), `src/swarm_arena/_exec/env.py` (`build_env` applies it)
- Modify: roadmap R4.7 (`Job.drop_env`), R5.4 item 4, the M4 **Must NOT** exception, R3.1 (`+M4` on `_exec/executor.py` and `_exec/env.py`)
- Test: `tests/_gateway/test_keys.py`, `tests/_exec/test_env.py`

**Interfaces:**
- Consumes: `GatewayAdapter`, `VirtualKey`, `SpendRow`, `GatewayError` (Task 11); `FakeGateway`, `LiteLLMProxy` (Task 12, tests); `PROVIDER_KEYS` (Task 3); `record_call` (R4.13.19); `JournalWriter`, `EventDraft`, `Event` (R4.13.3); `Reservation` (R4.13.6); `GatewayConfig`; `Job`, `build_env` (R4.7, M1a).
- Produces:

```python
# _exec/executor.py (+M4): Job gains
    drop_env: tuple[str, ...] = ()              # [plan] names build_env never takes from the node or the secrets file
# _exec/env.py (+M4): build_env removes every drop_env name after layer 2 (command.env and job.secret_env resolved
# from node_env or the secrets file); layers 3 to 5 (job.env, SWARM_ARENA_RESULT, job.secret_values) still apply, so
# the virtual key in secret_values reaches the tool as OPENAI_API_KEY and ANTHROPIC_API_KEY.

# _gateway/keys.py
REPLACED_KEYS: Final = frozenset({"OPENAI_API_KEY", "ANTHROPIC_API_KEY"})
def needs_key(providers: frozenset[str], secret_env: Sequence[str]) -> bool: ...   # providers, or a REPLACED_KEYS name
def unreplaced_provider_keys(secret_env: Sequence[str]) -> list[str]: ...          # sorted, in PROVIDER_KEYS - REPLACED_KEYS
def key_alias(candidate_id: str) -> str: ...                                       # f"cand-{candidate_id}"
def key_metadata(*, candidate_id: str, run_id: str, goal_id: str, team: str, agent: str) -> dict[str, Any]: ...
    # {candidate_id, run_id, goal_id, team, agent,
    #  "tags": ["swarm-arena", f"run:{run_id}", f"goal:{goal_id}", f"candidate:{candidate_id}"],
    #  "spend_logs_metadata": {candidate_id, run_id, goal_id, team, agent}}
def gateway_job(job: Job, *, url: str, secret: str) -> Job: ...
    # dataclasses.replace: env += OPENAI_BASE_URL=f"{url.rstrip('/')}/v1", ANTHROPIC_BASE_URL=url.rstrip('/');
    # secret_values += OPENAI_API_KEY=secret, ANTHROPIC_API_KEY=secret; secret_env -= PROVIDER_KEYS;
    # drop_env = tuple(sorted(PROVIDER_KEYS)), so a provider key that the tool's own Command.env allowlists
    # (for example a ScriptTool env entry) is not taken from the node either
class KeyRecord(FrozenModel):
    candidate_id: str; key_id: str; alias: str; reservation_id: str; max_budget_usd: float
    cap: Literal["hard", "soft"]; goal_id: str; team: str; agent: str; round: int
    closed_at: datetime | None = None
class KeyManager:
    def __init__(self, adapter: GatewayAdapter, config: GatewayConfig, *, cap: Literal["hard", "soft"],
                 journal: JournalWriter, settle_timeout_s: float) -> None: ...
    async def open(self, *, candidate_id: str, reservation: Reservation, run_id: str, goal_id: str, team: str,
                   agent: str, round: int) -> tuple[KeyRecord, str]: ...
        # create_key(alias, max_budget_usd=reservation.amount.usd, duration_s=config.key_duration_s, metadata,
        # default_estimated_output_tokens=config.default_estimated_output_tokens); then journal.append of a
        # gateway_key event {"action": "created", **record}. It is ordinary: the tool-run start commit covers it.
    async def close(self, record: KeyRecord) -> KeyRecord: ...
        # block_key; if closed_at(key) gives T: append gateway_key {"action": "closed", "key_id", "closed_at": T}
    async def revoke(self, record: KeyRecord) -> None: ...
        # block_key only: no closed_at read and no journal write (the outer-cancel path of Task 15; resume or
        # reconcile confirms the closure later)
    async def read(self, records: Sequence[KeyRecord], *, wait_first: bool) -> list[LLMCall]: ...
        # wait_first: read at once, then every min(1.0, settle_timeout_s) until a read has rows or settle_timeout_s
        # has passed. Each row → record_call(settle=False); duplicates (None) are left out of the result.
    def restore(self, events: Iterable[Event]) -> None: ...    # rebuild the records from committed gateway_key events
    def records(self, *, goal_id: str | None = None) -> list[KeyRecord]: ...
```

A row goes to `record_call` with `metadata={run_id, goal_id, team, agent, round, "role": "tool", "candidate_id"}`, `model=row.model`, `provider` = the text before the first `/` of `row.model` (else `"unknown"`), `usage=row.usage`, `cost_usd=row.cost_usd`, `cache_hit=False`, `latency_ms=None`, `status="ok"`, `error=None`, `request=None`, `response=None`, `litellm_call_id=row.litellm_call_id`, `tool_call_ids=()`, and `settle=False`.

- [ ] **Step 1: Write the failing tests**

```python
def test_needs_key_rules() -> None:
    assert needs_key(frozenset({"openai"}), ()) and needs_key(frozenset(), ["OPENAI_API_KEY"])
    assert not needs_key(frozenset(), ["GEMINI_API_KEY"]) and not needs_key(frozenset(), [])

def test_unreplaced_provider_keys() -> None:
    assert unreplaced_provider_keys(["MY_VAR", "OPENAI_API_KEY", "GEMINI_API_KEY"]) == ["GEMINI_API_KEY"]

def test_key_alias() -> None:
    assert key_alias("c-3f9a1c2b5d6e7f80-e1-0193") == "cand-c-3f9a1c2b5d6e7f80-e1-0193"

def test_key_metadata_tags_carry_candidate() -> None:
    meta = key_metadata(candidate_id="c-a-e1-0001", run_id="r1", goal_id="g", team="t", agent="a")
    assert "candidate:c-a-e1-0001" in meta["tags"] and meta["spend_logs_metadata"]["candidate_id"] == "c-a-e1-0001"

def test_gateway_job_env() -> None:
    job = gateway_job(_job(secret_env=("OPENAI_API_KEY", "GEMINI_API_KEY", "MY_VAR")), url="http://gw:4000/", secret="sk-v")
    assert job.env["OPENAI_BASE_URL"] == "http://gw:4000/v1" and job.env["ANTHROPIC_BASE_URL"] == "http://gw:4000"
    assert job.secret_values == {"OPENAI_API_KEY": "sk-v", "ANTHROPIC_API_KEY": "sk-v"} and job.secret_env == ("MY_VAR",)
    assert job.drop_env == tuple(sorted(PROVIDER_KEYS))

# tests/_exec/test_env.py
def test_drop_env_removes_node_values_not_secret_values(tmp_path: Path) -> None:
    job = make_job(drop_env=("GEMINI_API_KEY", "OPENAI_API_KEY"), secret_values={"OPENAI_API_KEY": "sk-v"})
    env = build_env(job, Command(argv=["x"], env=["GEMINI_API_KEY", "OPENAI_API_KEY"]), workspace=tmp_path,
                    node_env={"GEMINI_API_KEY": "real-g", "OPENAI_API_KEY": "real-o"})
    assert "GEMINI_API_KEY" not in env and env["OPENAI_API_KEY"] == "sk-v"

async def test_open_journals_record_without_secret(fake_gateway: FakeGateway, journal: JournalWriter) -> None:
    record, secret = await _manager(fake_gateway, journal).open(candidate_id="c-a-e1-0001", reservation=_res(1.0), ...)
    await journal.commit()
    assert record.alias == "cand-c-a-e1-0001" and record.max_budget_usd == 1.0
    assert secret.encode() not in _all_committed_bytes(journal)

async def test_read_waits_for_first_rows(fake_gateway: FakeGateway, journal: JournalWriter) -> None:
    manager, record = await _opened(fake_gateway, journal, settle_timeout_s=2.0)
    fake_gateway.add_spend(record.key_id, model="openai/gpt-6", cost_usd=0.02, tokens=(5, 1), call_id="r1",
                           visible_after_s=0.3)
    calls = await manager.read([await manager.close(record)], wait_first=True)
    assert [(c.role, c.candidate_id, c.litellm_call_id) for c in calls] == [("tool", "c-a-e1-0001", "r1")]

async def test_read_gives_up_after_settle_timeout(fake_gateway: FakeGateway, journal: JournalWriter) -> None:
    manager, record = await _opened(fake_gateway, journal, settle_timeout_s=0.5)
    start = time.monotonic()
    assert await manager.read([record], wait_first=True) == []
    assert 0.5 <= time.monotonic() - start < 1.5

async def test_rows_recorded_once(fake_gateway: FakeGateway, journal: JournalWriter) -> None:
    manager, record = await _opened_with_rows(fake_gateway, journal, call_ids=["r1"])
    assert len(await manager.read([record], wait_first=False)) == 1
    assert await manager.read([record], wait_first=False) == []

async def test_revoke_blocks_without_journal(fake_gateway: FakeGateway, journal: JournalWriter) -> None:
    manager, record = await _opened(fake_gateway, journal, settle_timeout_s=1.0)
    before = len(_appended(journal))                     # _appended: every event appended to the writer so far
    await manager.revoke(record)
    assert fake_gateway.keys[record.key_id]["blocked"] is True and len(_appended(journal)) == before

def test_restore_from_events(journal_events: list[Event]) -> None:   # one "created" and one "closed" event
    manager = KeyManager(...)
    manager.restore(journal_events)
    assert manager.records()[0].closed_at is not None
```

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/_gateway/test_keys.py tests/_exec/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._gateway.keys'` and `TypeError: ... unexpected keyword argument 'drop_env'`.

- [ ] **Step 3: Implement `Job.drop_env`, the `build_env` rule, and `_gateway/keys.py` with the signatures above.** `job_payload` (Task 5) serializes `drop_env` with the rest of the Job, so a Ray worker applies it too.

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/_gateway tests/_exec -v && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_gateway/keys.py src/swarm_arena/_exec/executor.py src/swarm_arena/_exec/env.py tests/_gateway/test_keys.py tests/_exec/test_env.py docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat: give each LLM-calling tool run its own gateway key" -m "KeyManager creates a cand-<id> key with max_budget equal to the run's reservation, points the tool at the gateway through Job.env and Job.secret_values, drops real provider keys, blocks the key after the run, and records spend rows with settle=False, once per LiteLLM call id. Job gains drop_env, which build_env applies after it resolves the allowlists, because the tool's own Command.env could otherwise pull a real provider key from the node (spec 7.3 item 2a step 2). Roadmap: R4.7, R5.4 item 4, and the M4 Must NOT exception."
```

---
### Task 14: Gateway preflight and the hard or soft cap label

**Files:**
- Modify: `src/swarm_arena/_engine/preflight.py` (`gateway_preflight`), `src/swarm_arena/_engine/run.py` (build the adapter and the `KeyManager` after preflight), `src/swarm_arena/_engine/manifest.py` (`llm_gateway` block), `src/swarm_arena/_board/trace.py` (key line)
- Modify: `docs/limits-and-safety.md` (section "LLM gateway: hard and soft caps"), roadmap R3.1 (`+M4` on `manifest.py`, `trace.py`, `run.py`)
- Modify: `tests/_engine/test_manifest.py::test_manifest_keys_and_no_secret_values` (M1b Task 25): add `"llm_gateway"` to the expected key set, because the manifest now always has that key
- Test: `tests/_engine/test_preflight.py`, `tests/_engine/test_manifest.py`, `tests/_board/test_trace.py`

**Interfaces:**
- Consumes: `adapter_for`, `LiteLLMProxy` (Task 12); `KeyManager` (Task 13); `GatewayConfig.hard_cap_required(limits)` (R4.8); `ProfileReport` (Task 11).
- Produces:

```python
# _engine/preflight.py (+M4)
async def gateway_preflight(config: RunConfig, adapter: GatewayAdapter) -> tuple[Literal["hard", "soft"], ProfileReport]: ...
    # verified → "hard". Not verified: if config.llm_gateway.hard_cap_required(config.limits) → ValueError, else "soft".
```

- Error text (exact): `f"LLM gateway strict profile not verified: {'; '.join(report.failures)}. Fix the gateway, or set llm_gateway.require_hard_cap = false to run with a soft cap."`
- `manifest.json` gains `"llm_gateway": {"url": str, "cap": "hard" | "soft", "verified": bool, "failures": list[str], "terminal_accounting": false}`, or `null` with no gateway. It never holds the master key or a key value. If M1b writes the manifest before preflight, move that write after preflight. It is still one create.
- `trace` for a keyed candidate (and so the board viewer's candidate page, which shows the same chain) adds `f"gateway key {alias}: max_budget {usd:.2f} USD, cap {cap}"`. After a reconcile, it adds `f"gateway spend {observed:.4f} USD ({label})"`, or `f"gateway spend {reserved:.2f} USD (estimated)"`. It reads the candidate's `gateway_key` and `reconcile` events.

- [ ] **Step 1: Write the failing tests**

```python
async def test_gateway_preflight_hard(fake_gateway: FakeGateway) -> None:   # _config: no max_usd
    cap, report = await gateway_preflight(_config(fake_gateway), _proxy(fake_gateway))
    if ABSENT_FAILURES:                                  # Task 11: no hard cap on this LiteLLM range
        assert cap == "soft" and report.verified is False and report.failures == ABSENT_FAILURES
    else:
        assert cap == "hard" and report.verified

async def test_gateway_preflight_fails_when_hard_cap_required() -> None:   # max_usd=5.0, require_hard_cap=None
    with FakeGateway(strict=False) as gw, pytest.raises(ValueError, match="strict profile not verified: "):
        await gateway_preflight(_config(gw, max_usd=5.0), _proxy(gw))

async def test_gateway_preflight_soft_when_not_required() -> None:
    with FakeGateway(strict=False) as gw:
        assert (await gateway_preflight(_config(gw, max_usd=5.0, require_hard_cap=False), _proxy(gw)))[0] == "soft"

def test_manifest_records_gateway_cap(...) -> None:
    assert _manifest(run_dir)["llm_gateway"] == {"url": gw.url, "cap": "soft", "verified": False,
                                                 "failures": failures, "terminal_accounting": False}

def test_manifest_without_gateway_is_null(...) -> None:
    assert _manifest(run_dir)["llm_gateway"] is None

def test_manifest_keys_and_no_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:   # M1b Task 25, updated
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-123")
    m = build_manifest(run_id="r", config=RunConfig(secrets=SecretsPolicy(allow=["ANTHROPIC_API_KEY"])), plans=[_plan()], capacity=None)
    assert set(m) == {"schema_version", "run_id", "created_at", "versions", "host", "executor_capacity",
                      "run_config_sha256", "plans", "tools", "llm_gateway"}           # + "llm_gateway" (null here)
    assert b"sk-secret-123" not in canonical_json(m)

def test_trace_shows_gateway_cap(...) -> None:     # journal with gateway_key created, cap "soft", max_budget 1.0
    assert "gateway key cand-c-a-e1-0001: max_budget 1.00 USD, cap soft" in _trace_text("c-a-e1-0001")
```

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/_engine/test_preflight.py tests/_engine/test_manifest.py tests/_board/test_trace.py -v`
Expected: FAIL with `ImportError: cannot import name 'gateway_preflight'`, with `KeyError: 'llm_gateway'`, and with an `AssertionError` on the key set in `test_manifest_keys_and_no_secret_values`.

- [ ] **Step 3: Implement. Write the docs section.** It covers: the strict profile list (spec §7.3 item 5, plus the per-key `max_tokens` bound and reserved routes only); `require_hard_cap` (default: true when `max_usd` is set); hard versus soft and where the label shows (manifest, `trace`, board viewer); USD caps only; pin the gateway's cost map. When Task 1 recorded any item as `absent`, the section names each absent rule and says that a hard cap is not available on this LiteLLM range, so `llm_gateway.require_hard_cap = false` is needed with `max_usd`.

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/_engine tests/_board -v && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_engine src/swarm_arena/_board/trace.py tests/_engine tests/_board docs
git commit -m "feat: check the strict gateway profile at preflight and label the cap" -m "A verified profile makes each key's max_budget a hard cap. Otherwise the run fails preflight when a hard cap is required (the default with max_usd), or goes on with a soft cap. The label is in manifest.json and in trace and the board viewer."
```

---

### Task 15: Keyed tool runs in the pipeline

**Files:**
- Modify: `src/swarm_arena/_agent/tool_run.py` (the keyed flow), `src/swarm_arena/_agent/runtime.py` (`AgentRuntime.gateway`)
- Modify: `src/swarm_arena/_budget/ledger.py` (`keep_charged` settle and replay)
- Modify: `src/swarm_arena/_engine/goal.py` (read back at round close), `src/swarm_arena/_engine/run.py` (read back at run end), `src/swarm_arena/_engine/resume.py` (restore keys, then read back)
- Modify: `src/swarm_arena/_planning/validate.py` (unreplaced-key warning), roadmap R4.13.10 and R3.1 tags. Roadmap R4.7 (`Job.drop_env`: with a gateway, every unkeyed tool job gets `REPLACED_KEYS`) and R5.4 item 18 (keying reads only the driver-visible allowlist, a deviation from §7.3 item 2a step 1) already hold this task's rules; check that they match, and name them in the commit body.
- Modify: `tests/_fakes/tools.py` (`EnvCommandTool(PyTool)`: no `providers()`; its `command()` runs `report_env.py` with `Command.env=["OPENAI_API_KEY"]`)
- Test: `tests/_budget/test_ledger.py`, `tests/_agent/test_tool_run.py`, `tests/_engine/test_goal.py`, `tests/_engine/test_resume.py`, `tests/_planning/test_validate.py`

**Interfaces:**
- Consumes: `KeyManager` (with `revoke`), `needs_key`, `gateway_job`, `unreplaced_provider_keys`, `REPLACED_KEYS` (Task 13); `llm_calls_source`, `record_reported`, `parse_llm_calls` (Task 3); `Ledger.reserve(..., keep_charged=True)`, `Ledger.settle` (R4.13.6); `GatewayError` (Task 11); `Fenced` (R4.13.1); `Estimate` (R4.2).
- Produces:
  - `AgentRuntime.gateway: KeyManager | None = None` (the last field; the engine sets it when `RunConfig.llm_gateway` is set).
  - Ledger: settling a `keep_charged` reservation charges the **full** `amount`. The settle event payload has `actual` (for reports), `charged` (= amount), and `keep_charged: true`. `end_goal` returns nothing of it, and `Ledger.replay` gives the same totals.
  - `gateway_key` event action `"read_failed"` with payload `{"action": "read_failed", "error": str}`. Read-back errors at round close, run end, and resume become this event, and they never stop the run.
  - `validate_plan` warning (exact), with a gateway: `f"agent {address}: tool allowlists {name}, which the LLM gateway does not replace; keyed tool runs do not get it"`.

Keyed flow inside `execute_tool_run` (the order is fixed by the spec and this plan):
1. `keyed = rt.gateway is not None and needs_key(tool.providers(), secret_env)`. `secret_env` is the allowlist that the driver can see before dispatch, the same input as Task 3's `may_call` and the M1b `secrets.allow` check: a `ScriptTool`'s `Config.env`, else `()`. A `Command.env` that `Tool.command()` builds on the executor is not seen (R5.4 item 18).
2. `reserve(..., keep_charged=keyed)`, as durable as in M1b.
3. When `rt.gateway` is set, an unkeyed tool job gets `drop_env = tuple(sorted(REPLACED_KEYS))`, so a `Command.env` allowlist can never take a real `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` from the node. A keyed job gets every `PROVIDER_KEYS` name from `gateway_job` (Task 13) in step 5.
4. If keyed, before `open`: a reservation with `amount.usd <= 0` (for example a tokens-only estimate) gets no key and is not dispatched. The output is `ToolOutput(status=ERROR)`, with the candidate warning `"LLM gateway key needs a USD reservation above 0; give Tool.estimate() a usd value or set limits.default_run_reservation.usd"`, and the reservation stays charged. Else `record, secret = await rt.gateway.open(...)`. On a `GatewayError`, do not dispatch. The output is `ToolOutput(status=ERROR)`, with the candidate warning `f"LLM gateway key creation failed: {err}"`, and the reservation stays charged.
5. Once step 4 has returned a key, steps 5 and 6 run inside `try:`. `job = gateway_job(job, url=..., secret=secret)` (it sets `Job.drop_env`, Task 13), then `submit`. The `except BaseException:` branch covers an outer-scope cancel, `Fenced` from the dispatch commit, and any error that leaves M1b's submit block: it runs `with anyio.CancelScope(shield=True, deadline=anyio.current_time() + 35.0): await rt.gateway.revoke(record)`, drops any `Exception` from `revoke`, and re-raises. Nothing is journaled on that path (M1b Task 19: after an outer cancel nothing more is committed); resume or reconcile confirms the closure later.
6. The normal path, a `run_scope` cancel included: in a shielded scope bounded by `move_on_after(settle_timeout_s + 30.0)`, `record = await close(record)`. If the run was not cancelled, also `calls = await read([record], wait_first=True)`. A `GatewayError` here adds the candidate warning `f"LLM gateway spend read-back failed: {err}; later reads and reconcile retry"`, and the result is kept.
7. Keyed run: `swarm/llm_calls.jsonl` rows are not recorded as `LLMCall` events (no `record_reported`). The file stays in `ToolOutput.llm_calls_file` for information only (spec §8.1: the gateway spend logs are the only source of actual spend), and its parse warnings are still added to `Candidate.warnings`. Unkeyed run: record the rows as in Task 3.
8. Set `Candidate.llm_call_ids` (keyed: only the gateway ids; unkeyed: the reported ids, Task 3) and `llm_calls_source = llm_calls_source(keyed=keyed, ...)`. Settle as in M1b; a keyed reservation stays fully charged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/_budget/test_ledger.py
async def test_keep_charged_settle_charges_full(ledger: Ledger) -> None:
    before = ledger.remaining("g/t/a")
    res = await ledger.reserve("g/t/a", Amount(usd=1.0), kind="tool", goal_id="g", keep_charged=True)
    await ledger.settle(res.id, Amount(usd=0.1))
    assert ledger.remaining("g/t/a") == before - Amount(usd=1.0)
    assert _last_settle(ledger)["charged"] == {"tokens": 0, "usd": 1.0} and _last_settle(ledger)["keep_charged"] is True

async def test_keep_charged_not_returned_at_end_goal(ledger: Ledger) -> None: ...        # run pool unchanged by end_goal
def test_replay_keep_charged_full(...) -> None: ...                                      # replayed remaining == live remaining

# tests/_agent/test_tool_run.py
async def test_keyed_run_gets_gateway_env(fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "real-provider-secret")
    c = await _run(tool=_report_env_tool(providers=["openai"], env=["OPENAI_API_KEY"]), gateway=fake_gateway)
    env = c.output.data["env"]
    assert env["OPENAI_BASE_URL"] == f"{fake_gateway.url}/v1" and env["ANTHROPIC_BASE_URL"] == fake_gateway.url
    assert env["OPENAI_API_KEY"].startswith("sk-") and env["OPENAI_API_KEY"] != "real-provider-secret"

async def test_keyed_run_never_gets_real_provider_keys(fake_gateway: FakeGateway, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "real-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "real-gemini")
    # a ScriptTool (report_env.py) whose own config env allowlists both names; secrets.allow has both
    c = await _run(tool=_report_env_tool(providers=["openai"], env=["OPENAI_API_KEY", "GEMINI_API_KEY"]),
                   gateway=fake_gateway, allow=["OPENAI_API_KEY", "GEMINI_API_KEY"])
    env = c.output.data["env"]
    assert "GEMINI_API_KEY" not in env and env["OPENAI_API_KEY"].startswith("sk-") and "real-openai" not in env.values()

async def test_unkeyed_run_has_no_gateway_env(fake_gateway: FakeGateway) -> None:
    c = await _run(tool=_report_env_tool(providers=[], env=[]), gateway=fake_gateway)
    assert "OPENAI_BASE_URL" not in c.output.data["env"] and not _calls(fake_gateway, "/key/generate")
    assert c.llm_calls_source == "none"

async def test_keyed_run_records_gateway_rows_and_keeps_reservation(fake_gateway: FakeGateway) -> None:
    c = await _run(tool=_chat_tool(calls=2), gateway=fake_gateway)                  # 2 calls to /v1/chat/completions
    assert c.llm_calls_source == "gateway" and len(c.llm_call_ids) == 2
    assert _settle_for(c)["charged"]["usd"] == _reserve_for(c)["amount"]["usd"]

async def test_keyed_run_reported_rows_are_information_only(fake_gateway: FakeGateway) -> None:
    # the tool reports its gateway call (the x-litellm-call-id, Task 12) with cost 9.0, plus one row with no call id
    c = await _run(tool=_chat_tool(calls=1, also_report_jsonl=True, reported_cost_usd=9.0, extra_reported_rows=1),
                   gateway=fake_gateway)
    assert len(_events("llm_call")) == 1 and _events("llm_call")[0].payload["call"]["cost_usd"] == 0.01   # gateway row
    assert c.output.llm_calls_file is not None and len(c.llm_call_ids) == 1

async def test_late_gateway_row_recorded_after_report(fake_gateway: FakeGateway) -> None:   # settle_timeout_s = 1.0
    fake_gateway.chat_visible_after_s = 1.5                       # the row shows after the run's own read gave up
    c = await _run(tool=_chat_tool(calls=1, also_report_jsonl=True, reported_cost_usd=9.0), gateway=fake_gateway)
    assert c.llm_call_ids == []                                   # the run's own read saw no row; the report is ignored
    await anyio.sleep(1.0)                                        # the row is visible now
    await _gateway().read(_gateway().records(), wait_first=False)  # the round-close read
    call_id = _chat_call_ids(fake_gateway)[0]
    calls = [e.payload["call"] for e in _events("llm_call") if e.payload["call"]["litellm_call_id"] == call_id]
    assert len(calls) == 1 and calls[0]["cost_usd"] == 0.01

async def test_unkeyed_tool_command_env_never_gets_real_replaced_key(fake_gateway: FakeGateway,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "real-o")
    c = await _run(tool=EnvCommandTool(), gateway=fake_gateway, allow=["OPENAI_API_KEY"])   # secrets.allow has it
    assert "OPENAI_API_KEY" not in c.output.data["env"] and not _calls(fake_gateway, "/key/generate")
    assert c.llm_calls_source == "none"

async def test_keyed_run_without_usd_reservation_not_dispatched(fake_gateway: FakeGateway) -> None:
    c = await _run(tool=_chat_tool(calls=1, estimate=Estimate(tokens=1000)), gateway=fake_gateway)
    assert c.output.status == RunStatus.ERROR and c.warnings[-1] == (
        "LLM gateway key needs a USD reservation above 0; give Tool.estimate() a usd value or set "
        "limits.default_run_reservation.usd")
    assert not _calls(fake_gateway, "/key/generate") and "tool_start" not in _types_for(c)

async def test_key_revoked_when_outer_scope_cancelled(fake_gateway: FakeGateway) -> None:
    with anyio.CancelScope() as outer:                            # cancelled once fake_gateway.keys holds the key
        async with anyio.create_task_group() as tg:
            tg.start_soon(_cancel_when, outer, lambda: bool(fake_gateway.keys))
            await _run(tool=_chat_tool(calls=1, sleep_s=60.0), gateway=fake_gateway)
    (key_id,) = fake_gateway.keys
    assert fake_gateway.keys[key_id]["blocked"] is True
    assert not [e for e in _appended_events("gateway_key") if e.payload["action"] == "closed"]

async def test_key_revoked_when_dispatch_commit_fenced(fake_gateway: FakeGateway) -> None:
    with pytest.raises(Fenced):                                   # the journal is fenced before dispatch
        await _run(tool=_chat_tool(calls=1), gateway=fake_gateway, fence_before_dispatch=True)
    (key_id,) = fake_gateway.keys
    assert fake_gateway.keys[key_id]["blocked"] is True and not _chat_call_ids(fake_gateway)   # the tool never ran

async def test_virtual_key_never_written(fake_gateway: FakeGateway, tmp_path: Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:                 # Review Focus 1
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))   # the local journal and workspaces go under tmp_path
    c = await _run(tool=_chat_tool(calls=1), gateway=fake_gateway, root=tmp_path)   # the tool itself never prints the key
    secret = _secret_of(fake_gateway, f"cand-{c.id}").encode()
    local = _local_journal_path()                            # <tempdir>/swarm-arena/<run-id>/e1.jsonl
    assert local.is_relative_to(tmp_path) and local.is_file() and secret not in local.read_bytes()
    assert all(secret not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())   # run_root, artifacts, local journal, logs
    assert secret.decode() not in repr(_last_job())

@pytest.mark.parametrize("fault", ["http_500", "slow"])
@pytest.mark.parametrize(("path", "dispatched", "status"), [
    ("/key/generate", False, RunStatus.ERROR), ("/key/block", True, RunStatus.OK), ("/spend/logs/v2", True, RunStatus.OK)])
async def test_gateway_failures_never_stop_the_run(fake_gateway: FakeGateway, path: str, dispatched: bool,
                                                   status: RunStatus, fault: str) -> None:        # Review Focus 3
    if fault == "http_500":
        fake_gateway.fail[path] = 500
    else:
        fake_gateway.delay[path] = 1.0                            # the test runtime's proxy has http_timeout_s=0.2
    first = await _run(tool=_chat_tool(calls=1), gateway=fake_gateway, http_timeout_s=0.2)
    assert first.output.status == status and ("tool_start" in _types_for(first)) is dispatched
    assert first.warnings and "LLM gateway" in first.warnings[-1]
    assert _settle_for(first)["charged"]["usd"] == _reserve_for(first)["amount"]["usd"]
    fake_gateway.fail.pop(path, None); fake_gateway.delay.pop(path, None)
    assert (await _run(tool=_chat_tool(calls=1), gateway=fake_gateway)).output.status == RunStatus.OK

# tests/_engine/test_goal.py
async def test_round_close_reads_late_rows(...) -> None:   # a row visible after the first read is recorded at round close
    assert [c.litellm_call_id for c in _llm_calls()] == ["early", "late"]

# tests/_engine/test_resume.py
async def test_resume_reads_back_open_keys(...) -> None:   # add_spend(..., cost_usd=0.02, call_id="late-after-crash")
    late = [c for c in _llm_calls() if c.litellm_call_id == "late-after-crash"]   # while the driver was down
    assert len(late) == 1 and late[0].cost_usd == 0.02                           # the gateway row's cost

async def test_resume_closes_keys_of_lost_runs(...) -> None:   # crashed journal: gateway_key "created" and tool_start,
    assert fake_gateway.keys[key_id]["blocked"]                 # no tool_end (the run is now run_lost)
    assert len([e for e in _events("gateway_key") if e.payload["action"] == "closed"
                and e.payload["key_id"] == key_id]) == 1

# tests/_planning/test_validate.py
def test_unreplaced_provider_key_warns() -> None:
    warnings = validate_plan(_plan(tool=ScriptTool(command=["python", "run.py"], env=["GEMINI_API_KEY"])),
                             _config(llm_gateway=GatewayConfig(url="http://gw:4000"), allow=["GEMINI_API_KEY"]))
    assert any("allowlists GEMINI_API_KEY, which the LLM gateway does not replace" in w for w in warnings)
```

Test helpers in `tests/_agent/test_tool_run.py`: `_run` builds its `JournalWriter` with `local_path` = the driver's local journal path (R4.13.3, M1b Tasks 26 and 27), `Path(tempfile.gettempdir()) / "swarm-arena" / run_id / f"e{epoch}.jsonl"`, and `_local_journal_path() -> Path` returns that path for the last `_run`. So `test_virtual_key_never_written` checks the same local file that a real run writes. `_run(..., http_timeout_s=30.0, fence_before_dispatch=False)` builds the runtime's `LiteLLMProxy` with that `http_timeout_s`, and with `fence_before_dispatch` fences the journal (another owner takes the lease) after the key is open, so the dispatch commit raises `Fenced`. `_gateway()` is the `KeyManager` of the last `_run`. `_chat_tool(calls, *, also_report_jsonl=False, reported_cost_usd=None, extra_reported_rows=0, sleep_s=0.0, estimate=None)` is a `_fakes.tools` tool with `providers=["openai"]` (and `estimate()` returning `estimate` when it is given) that posts `calls` chats to the gateway, can report them (and `extra_reported_rows` rows with no call id) in `swarm/llm_calls.jsonl`, then sleeps `sleep_s`. `_chat_call_ids(gw)` lists the call ids of the fake's chat rows. `_cancel_when(scope, cond)` polls `cond` every 0.05 s and cancels `scope`. `_appended_events(type)` reads every appended event, committed or not.

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/_budget tests/_agent tests/_engine tests/_planning -v`
Expected: FAIL. For example, `KeyError: 'charged'`, `KeyError: 'OPENAI_BASE_URL'`, and `assert 'none' == 'gateway'`.

- [ ] **Step 3: Implement the flow and the ledger rule above.** Round close (`goal.py`, after the round-close commit) calls `read(gateway.records(goal_id=goal), wait_first=False)`. Run end (`run.py`, before the final commit) and resume (`resume.py`, after `bind_run` and `restore(events)`) call `read(gateway.records(), wait_first=False)`. On resume, before that read: `close(record)` for every restored key that has no `closed_at` and whose candidate has no committed `tool_end` (it was in flight at the crash and is now `run_lost`); a `GatewayError` there becomes a `gateway_key` `"read_failed"` event.

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/_budget tests/_agent tests/_engine tests/_planning tests/_gateway -v && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_agent src/swarm_arena/_budget/ledger.py src/swarm_arena/_engine src/swarm_arena/_planning/validate.py tests docs/superpowers/plans/2026-10-18-swarm-arena-roadmap.md
git commit -m "feat: run LLM-calling tool runs through the gateway with their full reservation held" -m "Only a tool run that declares providers or allowlists a replaced provider key in its driver-visible config gets a key; with a gateway, unkeyed tool jobs drop OPENAI_API_KEY and ANTHROPIC_API_KEY. Its reservation stays fully charged for the whole run (spec 8.1), and a keyed run's self-reported rows are information only. Spend is read back after settle_timeout_s and again at round close, run end, and resume; a key is revoked when the run leaves by an outer cancel or a fenced commit; and gateway errors never stop the run. AgentRuntime gains the gateway field (roadmap R4.13.10). Roadmap R4.7 (Job.drop_env for unkeyed jobs) and R5.4 item 18 (keying reads only the driver-visible allowlist, a deviation from spec 7.3 item 2a step 1) record these rules."
```

---

### Task 16: Reconcile

**Files:**
- Create: `src/swarm_arena/_gateway/reconcile.py`
- Modify: `src/swarm_arena/_cli/cluster_cmds.py` (`reconcile`), `src/swarm_arena/_engine/run.py` (best-effort reconcile at run end), `src/swarm_arena/_exec/local.py` (add the start-time check to the M1a `LocalExecutor.confirm_dead`)
- Modify: `docs/limits-and-safety.md` (section "After the run: reconcile")
- Modify: roadmap R4.13.18 (add the CLI addressing rule: "[plan] Run-state commands (`status`, `stop`, `resume`, `reconcile`) take `RUN_DIR` (`<run_root>/<run-id>`) and, with no `--config`, use the run's own `run.toml`. Arena commands (M2: `review`, `leaderboard`, `board`, `learnings`, `trace`) take a run id or a namespace plus `--config PATH` (default `./run.toml`).")
- Modify: roadmap R4.7 (the `[plan, M4]` `LocalExecutor.confirm_dead` start-time rule; the roadmap already holds the text, so check that it matches this task and commit any fix with it)
- Test: `tests/_gateway/test_reconcile.py`, `tests/_cli/test_cluster_cmds.py`, `tests/_exec/test_local.py`, `tests/_engine/test_run.py`

**Interfaces:**
- Consumes: `KeyRecord`, `KeyManager.restore` (Task 13); `GatewayAdapter`, `ProfileReport` (Task 11); `adapter_for` (Task 12); `gateway_preflight` (Task 14); `check_storage(config: RunConfig) -> None` in `_engine/preflight.py` (M2 Task 2, R4.13.13) and `ConditionalWritesUnsupported` (R4.13.1); `Executor.confirm_dead` (R4.7, `RayExecutor` in Task 7); `open_store`, `read_owner`, `Lease.take_over` (with `expect_epoch`, M1b Task 3), `read_chain`, `JournalWriter`, `LeaseHeld`, `Fenced` (R4.13.1, R4.13.3; no core helper wraps the takeover retry of §7.1 step 1, so `take_run` writes its own loop, the same rule that M1b `resume_run` (Task 27) and M2 `HumanReview.import_csv` (Task 14) each write); the `takeover` payload shape of M1b `resume_run`, `{"force": bool}` (M1b Task 27 `test_seq_continues_and_takeover_logged`), which reconcile writes as `{"force": False}`; the driver's local journal path (R4.13.3, M1b Tasks 26 and 27), `Path(tempfile.gettempdir()) / "swarm-arena" / run_id / f"e{epoch}.jsonl"`, a local path even when `run_root` is `s3://`; `bind_run`, `unbind_run`, `record_call` (R4.13.19); `RunState.from_events` (R4.13.4), `IdGen(..., counters=...)`; `host_id`, `group_has_live_member`, `process_start_time` (R4.13.8).
- Produces:

```python
OBSERVED: Final = "observed"
UNVERIFIED: Final = "observed (completeness unverified)"
ESTIMATED: Final = "estimated"
class KeyReport(FrozenModel):
    candidate_id: str; key_id: str; alias: str; reserved_usd: float
    observed_usd: float | None                       # None unless every condition holds
    label: Literal["observed", "observed (completeness unverified)", "estimated"]
    reasons: list[str]                               # the failed conditions; [] when observed
    deleted: bool
class ReconcileReport(FrozenModel):
    run_id: str
    keys: list[KeyReport]
    def spend_usd(self) -> float: ...                # observed where observed, else reserved
async def reconcile_keys(events: Sequence[Event], *, journal: JournalWriter, adapter: GatewayAdapter, executor: Executor,
                         gateway: GatewayConfig, profile: ProfileReport | None,
                         now: Callable[[], datetime] = utc_now) -> ReconcileReport: ...
    # profile: the gateway's strict-profile report (its enforced max request duration and batch write interval);
    # None when the profile could not be read
async def take_run(config: RunConfig, run_id: str, *, owner_id: str,
                   force: bool) -> tuple[Lease, list[Event], JournalWriter]: ...
    # private to reconcile.py [plan]. First step, before the read_owner guard and the §7.1 loop:
    # await anyio.to_thread.run_sync(check_storage, config) (M2 Task 2; reconcile.py imports check_storage at module
    # level, so a test patches reconcile.check_storage); a ConditionalWritesUnsupported propagates and nothing is
    # written. Then store = open_store(config.run_root).child(run_id); the sync store and lease calls
    # run through anyio.to_thread.run_sync. If read_owner(store) is None → FileNotFoundError(f"run {run_id} has no
    # control/owner.json") (a guard for the CLI message only). Then take_run's own §7.1 step 1 loop (the same rule
    # as M1b resume_run, Task 27):
    #   loop:
    #       doc0 = read_owner(store)                          # the authority read
    #       try: lease = Lease.take_over(store, owner_id=owner_id, host=host_id(), ttl_s=config.lease_ttl_s,
    #                                    force=force, expect_epoch=doc0.epoch); break
    #       except Fenced: doc1 = read_owner(store)
    #           doc1.epoch == doc0.epoch → retry (the old owner renewed or advanced its head)
    #           else → raise Fenced(f"run {run_id}: another owner took the run (epoch {doc1.epoch})")
    #           (expect_epoch makes a takeover that lands between the authority read and take_over take this branch,
    #           as in M1b Task 3 and Task 27; without it, take_run would overwrite that takeover)
    # A LeaseHeld from Lease.take_over propagates. On LeaseHeld or Fenced, nothing is written.
    # Then events = the events of read_chain(store, lease.doc.head), oldest first (the run is fenced now, so the chain
    # is final); when lease.doc.head is None (no commit yet), events = [] and next_seq = 1.
    # writer = JournalWriter(store, lease, run_id=run_id, next_seq=head.seq + 1 (or 1),
    # local_path=Path(tempfile.gettempdir()) / "swarm-arena" / run_id / f"e{lease.doc.epoch}.jsonl" (the driver's
    # local journal path, R4.13.3 and M1b Tasks 26 and 27; local even when run_root is s3://),
    # sync_interval_s=config.sync_interval_s), not entered yet.
async def reconcile_run(config: RunConfig, run_id: str, adapter: GatewayAdapter, executor: Executor) -> ReconcileReport: ...
    # lease, events, writer = await take_run(config, run_id, owner_id=<new owner id>, force=False), where
    # LeaseHeld → RuntimeError(f"run {run_id} has a live owner; its own reconcile at run end covers the work") and
    # nothing is written, and a Fenced from take_run (another owner won the race) propagates. Then, in this order:
    #   async with writer:
    #       takeover event {"force": False} (the payload shape of M1b resume_run)
    #       state = RunState.from_events(events)
    #       bind_run(run_id, journal=writer, ids=IdGen(run_id, lease.doc.epoch, counters=state.id_counters),
    #                artifacts=ArtifactStore(open_store(config.artifact_root)), tracing=config.tracing,
    #                committed=list(state.llm_calls.values()), request_sha256=state.llm_request_sha256)
    #       try: profile = await adapter.preflight() (a GatewayError → None); reconcile_keys; commit
    #       finally: unbind_run(run_id)
    #   leaving the writer context releases the lease
# _exec/local.py (+M4): M1a confirm_dead returns `not group_has_live_member(pgid)` on this host; M4 adds the start-time check
async def confirm_dead(self, started: Job.Started) -> bool | None: ...
    # kind "process" on this host_id(): True if not group_has_live_member(pgid), or if procinfo.process_start_time(pid) is not
    # None and differs from started.start_time (the pgid now belongs to another process); False if alive with the
    # recorded (or an unknown) start time; None on another host or for kind "dispatch". local.py calls
    # procinfo.process_start_time through the module, so a test can patch it.
```

`reconcile_keys`, for each `gateway_key` "created" record (in this order, per §7.3):
1. **Condition 1:** `executor.confirm_dead(start)` is `True`. `start` comes from the candidate's `tool_start` or `tool_dispatch` event. A key with no start event counts as dead, because nothing ran [plan].
2. **Condition 2:** use the earliest journaled `closed_at`. If there is none, and the key is not deleted: call `block_key`, then `closed_at`. A confirmation is journaled as `gateway_key` "closed" with T.
3. **Condition 3:** the gateway enforces a request duration limit no longer than `gateway.request_timeout_s` (`profile` is not `None`, `profile.request_timeout_s` is not `None`, and `profile.request_timeout_s <= gateway.request_timeout_s`; this is the strict-profile duration rule of Task 11), the batch write interval is known (`profile.batch_write_interval_s` is not `None`), and `now() >= T + gateway.request_timeout_s + profile.batch_write_interval_s`. Without the duration limit, a request accepted before T could still run after the wait, so the wait proves nothing.
4. If all hold: `spend_rows` → record the new rows with `record_call(settle=False)`. `observed_usd` = the sum of `cost_usd` over all rows of the key. `label` = `OBSERVED` if `adapter.terminal_accounting`, else `UNVERIFIED`. Then `delete_key` and a `gateway_key` "deleted" event, unless the key is already deleted.
5. Else: `label = ESTIMATED`, `observed_usd = None`, and `reasons` lists the failed conditions with the exact texts `"tool not confirmed dead"`, `"key closure not confirmed by the gateway"`, `"gateway maximum request duration not verified"`, `"gateway batch write interval unknown"`, `"request_timeout_s has not passed since closure"`.
6. Journal one `reconcile` event: `{"keys": [KeyReport...], "terminal_accounting": bool}`. It never changes the ledger.

CLI: `swarm-arena reconcile RUN_DIR [--config RUN_TOML]`, addressed exactly like `resume` (M1b Task 28): `RUN_DIR` is `<run_root>/<run-id>`, the run id is its name, and with no `--config` it uses the run's own `run.toml`. It prints one row per key: `candidate`, `key alias`, `reserved USD`, `spend USD`, `label`. Exit 0. The live-owner `RuntimeError` from `reconcile_run` → its message on stderr and exit 1. At run end, the driver calls `reconcile_keys` with its own journal and the `ProfileReport` that `gateway_preflight` returned (Task 14) before the final commit. Any exception there becomes a `gateway_key` "read_failed" event, and the run still ends normally.

- [ ] **Step 1: Write the failing tests**

```python
async def test_all_conditions_met_is_observed_unverified(fake_gateway: FakeGateway, ...) -> None:
    clock = _Clock(T0)
    profile = _profile(request_timeout_s=300.0, batch_write_interval_s=10.0)      # ProfileReport, gateway limit 300 s
    report = await reconcile_keys(events, ..., profile=profile, now=clock)        # closes the key at T0
    clock.advance(600.0 + 10.0)
    report = await reconcile_keys(_events_now(), ..., profile=profile, now=clock)
    key = report.keys[0]
    assert key.label == "observed (completeness unverified)" and key.observed_usd == 0.03 and key.deleted
    assert fake_gateway.keys[key.key_id]["deleted"]

async def test_terminal_accounting_label_is_observed(...) -> None:            # adapter stub with terminal_accounting=True
    assert report.keys[0].label == "observed"

@pytest.mark.parametrize(("setup", "reason"), [
    ("alive", "tool not confirmed dead"), ("not_closed", "key closure not confirmed by the gateway"),
    ("too_soon", "request_timeout_s has not passed since closure"), ("no_interval", "gateway batch write interval unknown"),
    ("no_duration_limit", "gateway maximum request duration not verified"),     # profile.request_timeout_s is None
    ("duration_too_long", "gateway maximum request duration not verified"),     # 900 s > request_timeout_s 600 s
    ("no_profile", "gateway maximum request duration not verified")])           # profile=None (preflight failed)
async def test_failed_condition_is_estimated(setup: str, reason: str, ...) -> None:   # every other condition holds
    key = (await _reconcile(setup)).keys[0]
    assert key.label == "estimated" and key.observed_usd is None and reason in key.reasons and not key.deleted

async def test_closure_time_kept_from_first_confirmation(...) -> None:   # the second run uses the journaled T, not a later one
    assert _closed_events() == 1 and second.keys[0].label == "observed (completeness unverified)"

async def test_later_reconcile_updates_numbers(...) -> None:             # first "estimated", then observed after the wait
    assert [r.keys[0].label for r in (first, second)] == ["estimated", "observed (completeness unverified)"]

async def test_reconcile_run_fails_with_live_owner_writes_nothing(tmp_path: Path, ...) -> None:
    before = (run_dir / "control" / "owner.json").read_bytes()
    with pytest.raises(RuntimeError, match="has a live owner"):           # from LeaseHeld
        await reconcile_run(config, run_id, proxy, LocalExecutor())
    assert (run_dir / "control" / "owner.json").read_bytes() == before and _segments(run_dir) == segments_before

async def test_reconcile_checks_storage_before_takeover(tmp_path: Path, ..., monkeypatch: pytest.MonkeyPatch) -> None:
    # a finished run with an expired lease, so without the probe the takeover would succeed
    def refuse(config: RunConfig) -> None:
        raise ConditionalWritesUnsupported("s3://bucket: conditional writes not honoured")
    monkeypatch.setattr(reconcile, "check_storage", refuse)
    before, types_before = (run_dir / "control" / "owner.json").read_bytes(), _committed_types(run_dir)
    with pytest.raises(ConditionalWritesUnsupported):
        await reconcile_run(config, run_id, proxy, LocalExecutor())
    assert (run_dir / "control" / "owner.json").read_bytes() == before and _segments(run_dir) == segments_before
    assert _committed_types(run_dir) == types_before            # no new takeover or reconcile event: the head is unchanged

async def test_reconcile_run_takes_lease_commits_releases(...) -> None:
    owner = _owner(run_dir)
    assert owner.epoch == old_epoch + 1 and owner.expires_at < utc_now() and "reconcile" in _committed_types(run_dir)
    assert _takeover_payloads(run_dir) == [{"force": False}]          # the M1b resume_run shape (Task 27)

async def test_reconcile_run_unbound_records_new_row_once(fake_gateway: FakeGateway, tmp_path: Path) -> None:
    run_id = await _finished_keyed_run(fake_gateway, tmp_path)       # Swarm.run has returned: the run is unbound
    fake_gateway.add_spend(_key_id(fake_gateway), model="openai/gpt-6", cost_usd=0.02, tokens=(5, 1), call_id="late-1")
    config = _config(tmp_path, request_timeout_s=0.2)                 # FakeGateway enforces 0.1 s
    await anyio.sleep(0.5)
    first = 0 if MAX_REQUEST_DURATION is None else 1                  # Task 1 item 3 absent: condition 3 never holds
    for expected_new in (first, 0):                                   # the second reconcile records nothing new
        before = _committed_llm_call_ids(tmp_path, run_id)
        await reconcile_run(config, run_id, _proxy(fake_gateway), LocalExecutor())
        new = [i for i in _committed_llm_call_ids(tmp_path, run_id) if i not in before]
        assert len(new) == expected_new and (expected_new == 0 or _litellm_id(tmp_path, run_id, new[0]) == "late-1")
        with pytest.raises(ValueError, match="is not bound"):         # reconcile_run unbound the run again
            await record_call(metadata={"run_id": run_id, "round": 1}, **_ROW_FIELDS)

async def test_reconcile_run_without_head_reports_no_keys(tmp_path: Path, ...) -> None:   # owner.json, no commit yet
    report = await reconcile_run(config, run_id, proxy, LocalExecutor())
    assert report.keys == [] and _owner(run_dir).expires_at < utc_now()            # the lease is released

async def test_reconcile_run_retries_when_old_owner_writes_meanwhile(..., monkeypatch: pytest.MonkeyPatch) -> None:
    real = Lease.take_over.__func__
    attempts: list[int] = []
    def racing(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        attempts.append(1)
        if len(attempts) == 1:
            _bump_owner_version(store)            # same epoch, version + 1: the old owner wrote between read and replace
            raise Fenced("another owner changed the lease")
        return real(cls, store, **kw)
    monkeypatch.setattr(Lease, "take_over", classmethod(racing))
    await reconcile_run(config, run_id, proxy, LocalExecutor())
    assert len(attempts) == 2 and _owner(run_dir).epoch == old_epoch + 1    # §7.1: take_run retried from the read

async def test_reconcile_run_concurrent_owner_raises_fenced(..., monkeypatch: pytest.MonkeyPatch) -> None:
    real = Lease.take_over.__func__
    def other_first(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        real(cls, store, **{**kw, "owner_id": "other"})                     # another owner wins: epoch + 1
        raise Fenced("another owner changed the lease")
    monkeypatch.setattr(Lease, "take_over", classmethod(other_first))
    with pytest.raises(Fenced, match="another owner took the run"):        # from take_run: another epoch, no retry
        await reconcile_run(config, run_id, proxy, LocalExecutor())
    assert _owner(run_dir).owner_id == "other" and _segments(run_dir) == segments_before

async def test_reconcile_run_after_a_concurrent_takeover_raises_fenced(..., monkeypatch: pytest.MonkeyPatch) -> None:
    real = Lease.take_over.__func__
    calls: list[int] = []
    def takeover_lands_first(cls: type[Lease], store: Store, **kw: Any) -> Lease:
        calls.append(1)
        if len(calls) == 1:                                    # lands between the authority read and take_over
            real(cls, store, owner_id="other", host="h", ttl_s=60, force=True)
        return real(cls, store, **kw)                          # take_run's own call, with expect_epoch=old_epoch
    monkeypatch.setattr(Lease, "take_over", classmethod(takeover_lands_first))
    with pytest.raises(Fenced, match=f"run {run_id}: another owner took the run"):
        await reconcile_run(config, run_id, proxy, LocalExecutor())
    doc = _owner(run_dir)
    assert (doc.owner_id, doc.epoch) == ("other", old_epoch + 1)            # not + 2: take_run wrote nothing
    assert _segments(run_dir) == segments_before

# tests/_exec/test_local.py
async def test_confirm_dead_start_time_mismatch_is_dead(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        started = Job.Started(kind="process", job_id="x", host_id=host_id(), pid=proc.pid, pgid=proc.pid,
                              start_time=process_start_time(proc.pid))
        assert await LocalExecutor().confirm_dead(started) is False                # alive, same start time
        monkeypatch.setattr(procinfo, "process_start_time", lambda pid: "Thu Jan  1 00:00:00 1970")
        assert await LocalExecutor().confirm_dead(started) is True                 # the pgid is another process now
        assert await LocalExecutor().confirm_dead(Job.Started(kind="dispatch", job_id="x", task_name="x")) is None
    finally:
        proc.kill(); proc.wait()

# tests/_cli/test_cluster_cmds.py
def test_reconcile_cmd_prints_labels(...) -> None:          # fake gateway, a finished run; run_dir = <run_root>/<run-id>
    assert main(["reconcile", str(run_dir), "--config", str(run_toml)]) == 0 and "estimated" in capsys.readouterr().out

def test_reconcile_cmd_live_owner_exits_1(...) -> None:     # no --config: the run's own run.toml, as resume does
    assert main(["reconcile", str(run_dir)]) == 1 and "has a live owner" in capsys.readouterr().err

# tests/_engine/test_run.py
async def test_run_end_reconcile_is_best_effort(...) -> None:   # the gateway fails every call at run end
    assert result["g"] and "read_failed" in {e.payload["action"] for e in _events("gateway_key")}
```

Test helper in `tests/_gateway/test_reconcile.py`: `_bump_owner_version(store: Store) -> None` reads `control/owner.json` with `read_owner` and writes it back with `store.replace(OWNER_KEY, ..., if_match=etag)`, `version + 1`, the same epoch and head (the old owner's renew, as M1b Task 27 `test_takeover_retries_when_old_owner_commits_meanwhile` does with `renew()`). `_takeover_payloads(run_dir: Path) -> list[dict[str, Any]]` returns the payloads of the committed `takeover` events. `_finished_keyed_run` runs one keyed tool run to the end with `Swarm.run` against the fake gateway and returns the run id. `_committed_llm_call_ids` lists the `LLMCall` ids of the committed `llm_call` events, and `_litellm_id` gives one call's `litellm_call_id`. `_ROW_FIELDS` holds the other `record_call` keyword arguments of one row. `reconcile` is the module `swarm_arena._gateway.reconcile`. The three race tests patch `Lease.take_over`, which `take_run` calls, so they test the `take_run` loop itself: a `Fenced` with the same epoch retries from the authority read, a `Fenced` with another epoch propagates, and a takeover that lands between the authority read and `take_over` is fenced by `expect_epoch` (as M1b Task 3 `test_take_over_with_stale_expected_epoch_is_fenced` and Task 27 `test_resume_after_a_concurrent_takeover_fails_cleanly`) and is never overwritten.

- [ ] **Step 2: Run the tests to check that they fail**

Run: `uv run pytest tests/_gateway/test_reconcile.py tests/_cli/test_cluster_cmds.py tests/_exec/test_local.py tests/_engine/test_run.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'swarm_arena._gateway.reconcile'`, and with `assert False is True` for the start-time mismatch (M1a `confirm_dead` checks only `group_has_live_member`).

- [ ] **Step 3: Implement with the signatures and the order above. Write the docs section.** It covers the three conditions, the labels, why an observed number is never proof (late log batches), that reconcile never changes a live run's budget, that it probes `s3://` roots for conditional writes (`check_storage`) before it takes over, that it takes the run lease, and that a later `reconcile` updates the numbers.

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/_gateway tests/_cli tests/_exec tests/_engine -v && uv run ty check`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/swarm_arena/_gateway/reconcile.py src/swarm_arena/_cli/cluster_cmds.py src/swarm_arena/_engine/run.py src/swarm_arena/_exec/local.py tests docs
git commit -m "feat: add swarm-arena reconcile for gateway spend after the run" -m "Records a key's observed spend only when the tool is confirmed dead, the gateway confirms the key closed at T, the gateway enforces a request duration limit no longer than request_timeout_s, and request_timeout_s plus one batch interval has passed since T; otherwise the full reservation stays, labelled estimated. It probes s3:// roots with check_storage first, then takes the run lease with the takeover retry of spec 7.1 step 1 (the same rule as resume and review import), journals a takeover event with the resume payload shape (force false), binds the run for record_call from its replayed state and unbinds it after, handles a run with no commit yet, and runs best-effort at run end. LocalExecutor.confirm_dead also treats a pgid with another start time as dead (roadmap R4.7). Roadmap R4.13.18: one CLI addressing rule; reconcile takes RUN_DIR, as resume does."
```

---
### Task 17: Gateway end to end

**Files:**
- Create: `tests/e2e/test_gateway_run.py`
- Modify: `tests/_fakes/tools.py` (`llm_tool(prompt: str) -> dict`, which posts to `$OPENAI_BASE_URL/chat/completions` with `$OPENAI_API_KEY` through `urllib`, and `plain_tool(x: float) -> dict`)
- Modify: `docs/running-on-ray.md` (section "The LLM gateway on the cluster"), `docs/writing-a-connector.md` (a tool that calls an LLM declares `providers()`), `README.md` (one paragraph), `CHANGELOG.md`

**Interfaces:**
- Consumes: everything from Tasks 11 to 16; `Swarm.run`, `ScriptedManager`, `FunctionTool` (with `providers`), `main` (CLI).
- Produces: the M4 exit test `test_gateway_run.py`.

The plan has one goal, two teams, and one round. Team `llm` uses `FunctionTool("_fakes.tools:llm_tool", providers=["openai"])`, and team `plain` uses `FunctionTool("_fakes.tools:plain_tool")`. Both use `ScriptedManager` with 2 requests. `RunConfig`: `llm_gateway = GatewayConfig(url=fake.url)`, `settle_timeout_s = 1.0`, `limits.max_usd = 5.0`, `limits.max_tool_runs_per_round = 2`, `limits.default_run_reservation = Estimate(tokens=100_000, usd=0.5)`, and the master key env set in the test. When `ABSENT_FAILURES` (Task 11) is not empty, the strict fake is unverified, so `_run_swarm`'s default is `llm_gateway.require_hard_cap = False`; `test_hard_cap_required_fails_preflight` passes `require_hard_cap=None` and stays as it is. The tools have no `estimate()`, so the M1b worst-case estimate is 2 agents × 1 round × 2 runs × 0.50 USD / 0.85 = 2.35 USD, under `max_usd`, and `Swarm.run` reaches preflight. (With the defaults, 3 runs × 1.00 USD, it would be 7.06 USD and fail before preflight.)

- [ ] **Step 1: Write the failing tests**

```python
async def test_estimate_fits_max_usd(fake_gateway: FakeGateway, tmp_path: Path) -> None:
    run_id = await _run_swarm_id(fake_gateway, tmp_path)
    usd = _run_started(tmp_path, run_id)["estimate"]["usd"]             # the M1b run_started payload key names
    assert usd == pytest.approx(2 * 1 * 2 * 0.5 / 0.85) and usd < 5.0

async def test_key_per_llm_tool_run_only(fake_gateway: FakeGateway, tmp_path: Path) -> None:
    candidates = await _run_swarm(fake_gateway, tmp_path)
    llm = [c for c in candidates if c.team == "llm"]
    plain = [c for c in candidates if c.team == "plain"]
    assert sorted(_aliases(fake_gateway)) == sorted(f"cand-{c.id}" for c in llm)
    assert all(c.llm_calls_source == "gateway" and c.llm_call_ids for c in llm)
    assert all(c.llm_calls_source == "none" and not c.llm_call_ids for c in plain)

async def test_full_reservation_held(fake_gateway: FakeGateway, tmp_path: Path) -> None:
    candidates = await _run_swarm(fake_gateway, tmp_path)
    for c in (c for c in candidates if c.team == "llm"):
        assert _settle_for(tmp_path, c)["charged"] == _reserve_for(tmp_path, c)["amount"]

async def test_reconcile_labels(fake_gateway: FakeGateway, tmp_path: Path) -> None:
    run_id = await _run_swarm_id(fake_gateway, tmp_path)                 # request_timeout_s = 600 during the run
    assert {k["label"] for k in _last_reconcile(tmp_path, run_id)["keys"]} == {"estimated"}
    run_toml = _write_config(tmp_path, request_timeout_s=0.2)            # FakeGateway enforces 0.1 s <= 0.2 s (Task 12)
    await anyio.sleep(0.5)
    # main calls anyio.run, which cannot run inside this test's event loop, so it runs in a worker thread
    run_dir = _run_dir(tmp_path, run_id)                                 # <run_root>/<run-id>: reconcile takes RUN_DIR
    assert await anyio.to_thread.run_sync(main, ["reconcile", str(run_dir), "--config", str(run_toml)]) == 0
    keys = _last_reconcile(tmp_path, run_id)["keys"]
    if MAX_REQUEST_DURATION is None:                                     # Task 1 item 3 absent: condition 3 never holds
        assert {k["label"] for k in keys} == {"estimated"}
        assert not any(fake_gateway.keys[k["key_id"]]["deleted"] for k in keys)
        return
    assert {k["label"] for k in keys} == {"observed (completeness unverified)"}
    assert all(fake_gateway.keys[k["key_id"]]["deleted"] for k in keys)
    assert sum(k["observed_usd"] for k in keys) == pytest.approx(_fake_spend_total(fake_gateway))

async def test_soft_cap_label(tmp_path: Path) -> None:
    with FakeGateway(strict=False) as gw:
        await _run_swarm(gw, tmp_path, require_hard_cap=False)
        assert _manifest(tmp_path)["llm_gateway"]["cap"] == "soft"
        assert "cap soft" in _trace_text(tmp_path, _first_llm_candidate(tmp_path))

async def test_hard_cap_required_fails_preflight(tmp_path: Path) -> None:
    with FakeGateway(strict=False) as gw, pytest.raises(ValueError, match="strict profile not verified"):
        await _run_swarm(gw, tmp_path, require_hard_cap=None)   # max_usd set; the 2.35 USD estimate passes
    assert not _calls(gw, "/key/generate")
```

- [ ] **Step 2: Run the tests to check that they fail, or fix what they find**

Run: `uv run pytest tests/e2e/test_gateway_run.py -v`
Expected: FAIL at first with `ModuleNotFoundError` for the new `_fakes.tools` functions. After you add them, any failure that is left is an integration bug in Tasks 11 to 16. Fix it in the owning module, and add a unit test there.

- [ ] **Step 3: Write the docs.** "The LLM gateway on the cluster" covers:
  - The owner runs LiteLLM Proxy with Postgres and a master key; the `run.toml` `[llm_gateway]` example.
  - Which tool runs get a key, and the variables they get. A tool that calls an LLM must declare `providers()`: with a gateway, a `Command.env` allowlist alone never gets the real OpenAI or Anthropic key (R5.4 item 18). `docs/writing-a-connector.md` says the same.
  - A keyed run needs a USD reservation above 0 (`Tool.estimate()` with `usd`, or `limits.default_run_reservation.usd`).
  - Secret values in `Job.secret_values` (the gateway virtual keys) travel in the task payload, which Ray's object store can spill to worker-local disk; secrets that must never touch disk come from the worker's environment or `secrets.file`.
  - The full-reservation trade-off (a tighter `Tool.estimate()` and a correct `Tool.providers()` help).
  - The gateway's own `cache` and `langfuse` callback settings for tool-internal calls.
  - The network rule: block outbound traffic from workers to provider hosts, so a tool with a hard-coded provider URL cannot go around the gateway.
  - `swarm-arena reconcile`.

  README: one paragraph and a link. CHANGELOG: an `Unreleased` entry.

- [ ] **Step 4: Run the tests to check that they pass**

Run: `uv run pytest tests/_gateway tests/e2e/test_gateway_run.py -v && uv run sphinx-build -W --keep-going -b html docs docs/_build/html`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e/test_gateway_run.py tests/_fakes/tools.py docs README.md CHANGELOG.md src
git commit -m "test: add the gateway end-to-end run and document the gateway on the cluster" -m "One run with an LLM-calling tool and a plain tool proves: a key only for LLM tool runs, the full reservation held, and the estimated label at run end changing to observed (completeness unverified) after reconcile. The soft-cap label and the hard-cap preflight failure are also covered."
```

---

### Task 18: Milestone exit

**Files:**
- Modify: `CHANGELOG.md` (check the `Unreleased` M4 entries), `tests/test_public_api.py` (only if a snapshot is missing)

**Interfaces:**
- Consumes: everything above.
- Produces: `main` with M4 merged and every exit criterion green.

- [ ] **Step 1: Run the common gate (R2.0)**

```bash
uv sync --all-extras --all-groups
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Expected: each command exits 0. Default `pytest` collects no `ray`, `s3`, `llm_live`, or `gateway_live` test.

- [ ] **Step 2: Run the M4 exit criteria**

```bash
uv run pytest -m "ray and not s3" -v
uv run pytest tests/_gateway tests/e2e/test_gateway_run.py -v
uv run pytest tests/e2e/test_ray_cancel.py tests/e2e/test_ray_dispatch_crash.py tests/e2e/test_rosenbrock_ray.py -m "ray and not s3" -v
uv run pytest tests/ray/test_executor.py -m "ray and s3" -v     # skips unless SWARM_ARENA_TEST_S3_ENDPOINT is set
```

Expected: all PASS. `test_ray_cancel.py`: a soft cancel gives a partial output, and a force cancel kills grandchildren. `test_ray_dispatch_crash.py`: resume finds the task by name, and the run is `lost`. `test_rosenbrock_ray.py`. `test_gateway_run.py`: a key per LLM tool run only, the full reservation held, and the reconcile labels.

- [ ] **Step 3: Check that the core works with no extras (the `core` CI job)**

```bash
UV_PROJECT_ENVIRONMENT=/tmp/sa-core uv sync --all-groups
UV_PROJECT_ENVIRONMENT=/tmp/sa-core uv run --no-sync pytest tests/test_import_isolation.py tests/_gateway tests/_cli -v
```

Expected: PASS. The `ray` test paths are not collected (`collect_ignore`), and `swarm-arena --help` loads no `ray`.

- [ ] **Step 4: Check the Review Focus tests by name**

Run: `uv run pytest "tests/_agent/test_tool_run.py::test_virtual_key_never_written" "tests/_agent/test_tool_run.py::test_gateway_failures_never_stop_the_run" "tests/_tooling/test_llm_calls.py::test_hostile_llm_calls_file_never_crashes" -v && uv run pytest "tests/ray/test_executor.py::test_cancel_while_pending_never_runs_tool" "tests/_engine/test_resume.py::test_resume_dispatch_never_submitted_is_lost" -m ray -v`
Expected: 10 PASS (the parametrized test counts 6).

- [ ] **Step 5: Manual check (not in CI, with the owner's gateway)**

Run: `SWARM_ARENA_LIVE_GATEWAY_URL=<url> LITELLM_MASTER_KEY=<key> uv run pytest -m gateway_live -v`
Expected: PASS, or a profile failure list that matches the gateway's real settings. Record the result in the PR description.

- [ ] **Step 6: Commit, push, and open the PR (after the owner's go-ahead, because it reaches an audience)**

```bash
git add CHANGELOG.md tests/test_public_api.py
git commit -m "chore: close milestone M4 Cluster" -m "All M4 exit criteria pass: the common gate, pytest -m 'ray and not s3', and the gateway tests against the stdlib fake gateway."
git push -u origin m4-cluster
gh pr create --title "M4 Cluster: RayExecutor, LLM gateway, reconcile" --body "Implements docs/superpowers/plans/2026-10-18-swarm-arena-m4-cluster.md. Exit criteria: common gate, pytest -m 'ray and not s3', tests/_gateway and tests/e2e/test_gateway_run.py. gateway_live result: <result>. Strict-profile rules absent in LiteLLM 1.105.x (always unverified, so no hard cap): <the ABSENT_FAILURES texts, or none>."
```

The PR body has no AI attribution text. It lists every absent rule from Task 1 (the `ABSENT_FAILURES` texts), or says none.
