# Limits and safety

`RunConfig` and `Limits` are human-controlled. Plans and managers cannot raise
them. See the [public API](api/swarm_arena.md),
[configuration API](api/config.md), [manager API](api/manager.md), and
[executor API](api/executor.md).

## Defaults

| Limit | Default |
|---|---|
| `max_teams` per goal, including parts | 4 |
| `max_agents_per_team` | 4 |
| `max_agents_per_goal` | 12 |
| `max_concurrent_goals` | 4 |
| `max_agents_total` working concurrently | 32 |
| `max_agents_per_fingerprint` | 2 |
| `max_rounds` | 3 |
| `max_turns_per_round` per agent | 20 |
| `max_tool_runs_per_round` per agent | 3 |
| `max_tokens` / `max_usd` | Unset: no cap in that unit |
| `default_run_reservation` for a tool without an estimate | 100,000 tokens / 1.00 USD |
| `llm_tranche_tokens` / `llm_tranche_usd` | 25,000 tokens / 0.25 USD |
| `tool_timeout_s` | 3,600 s |
| `round_timeout_s` | Stored as `None`; effective default is 4 × tool timeout (14,400 s) |
| `run_timeout_s` | 86,400 s (24 h of active driver time) |
| `max_learnings_per_agent_round` | 3; zero disables publishing |
| `max_board_reads_per_round` | 10; zero disables reads |
| `digest_tokens` | 1,500 UTF-8 bytes |
| `max_edit_bytes` per submitted file | 262,144 (256 KiB) |
| `max_notes_bytes` | 16,384 (16 KiB) |
| `max_consecutive_failures` | 3 |
| `max_refused_repeats` | 3 allowed refusals; stops on the fourth |
| `watchdog_idle_rounds` | 2 |

Despite its name, `digest_tokens` counts **UTF-8 bytes**. This is a conservative
hard bound on tokens for every model; English text typically uses about a
quarter as many tokens. The digest is truncated to that byte budget.

`RunConfig.reserve_fraction` defaults to 0.15. `lease_ttl_s`, `sync_interval_s`,
and `settle_timeout_s` each default to 60 s; the lease renews every
`lease_ttl_s / 3`. `schedule_timeout_s` defaults to 600 s. `ConvergeSpec`
defaults are `top_n=2`, `parents_per_part=1`, and `max_join_attempts=1`.
Learning titles hold at most 120 characters and text at most 1,500 characters.
The `llm` extra provides built-in model integrations and checks used aliases,
role permissions, manager function calling, output caps, and prices under a
dollar cap; structured roles without response-schema support receive a warning
and use the two-stage fallback.

The optional `Limits.max_usd` run cap must be positive and no greater than
1,000,000,000 USD. Leaving it unset means there is no configured dollar cap.
Budget balances still support only the inclusive range from -1,000,000,000
to 1,000,000,000 USD, preserving exact accounting in whole micro-dollars.
Construction and arithmetic refuse amounts outside that range. Allocations
round down to whole micro-dollars, with the remainder retained in the parent pool.

```python
from swarm_arena import Limits, RunConfig

config = RunConfig(limits=Limits(max_usd=100.0, tool_timeout_s=120))
assert config.limits.effective_round_timeout_s() == 480.0
assert config.reserve_fraction == 0.15
```

## Approval and budget allocation

Before publishing run snapshots, `Swarm.run` admits plans into temporary artifact
storage and estimates work using effective tools and empty future inputs.
It rejects an estimate over a configured cap or a reserve too small for all
final slices. It constructs the executor, checks capacity, and passes the numeric
estimate and warnings to the async `approve` callback. `False` refuses without
writing run or artifact roots; temporary snapshots are discarded. `approve=None`
accepts an estimate within the limits. After approval, component preflight runs
and the approved snapshots are promoted without rereading their sources.

The startup error boundary covers executor construction, capacity results, and
approval callbacks. Diagnostics expose safe identifiers or exception types,
never callback exception values. Cancellation and lost ownership retain their
control meaning. The CLI likewise renders safe core diagnostics or a type name.

The budget is allocated as follows:

1. Hold `reserve_fraction` at run level for judging, replanning, convergence,
   and final judging. The configured-curator seam and built-in `LLMCurator`
   also use this reserve.
2. Hold a protected **final slice for every goal**, including queued goals.
   It covers one convergence plus `max_join_attempts` additional attempts and
   the final ranking batch's output terms; serialized inputs are reserved at
   dispatch and can still be refused (see [LLM-judge funding](scoring-and-review.md#llm-judges)).
   Converge tools are estimated at **every
   reachable round from 1 through R + 1**, taking the maximum in each budget
   unit, because a stop can converge before all R ordinary rounds finish.
   Ordinary judging and replanning cannot consume these protected slices.
3. Divide work among goals using `goal_weights` (equal weights by default).
4. At each round start, divide the remaining goal pool by the number of
   remaining rounds.
5. Divide a round's pool equally among active agents. An unused agent share
   returns to the goal for later rounds, rather than another agent in that round.
   Pruning therefore increases survivors' later shares.

Ordinary judges first use shared reserve, then the goal's work pool. If neither
can pay, the goal records `reserve_exhausted` and goes to select-only convergence.
When a goal ends, its unused work is redistributed to unfinished goals by their
weights, or retained as unallocated work if none remain.

## Committed reservations and crash recovery

Before dispatching a tool run, the core commits its estimated reservation to
the run journal. `Tool.estimate()` returning `None` uses
`default_run_reservation`; an explicit zero estimate means zero. Concurrent
reservations cannot overdraw the same share. Commits may group reservations
within a 0.2 s window, but work waits for the committed head.

Settlement is idempotent by reservation ID. Ordinary reported usage releases
unused funding back to its original pools. After a committed start,
provider-using tools retain the full reservation when usage cannot be recovered,
including invalid output; available usage otherwise settles the reported amount.
`settle_timeout_s` bounds the wait for pending call reports and post-stop or
post-deadline settlement, including durable stop-result publication. Healthy
live settlement remains under its owner's supervision without this fixed timer.
Report arrival has its own timeout, including after a graceful stop; expiry
records a conservative charge rather than a storage-cleanup failure.
Without a committed settle, the reservation stays charged in full. Cancelling construction or command preparation
before a committed start releases the reservation at zero while ownership is
held; a late preparation result never launches or journals work.

On resume, every reservation without a committed settlement stays charged in
full. Runs with a committed completion are recovered; runs with a committed
start but no committed completion become `lost`, and retrying them needs new
budget. Reservations for dispatches without a committed start remain charged
conservatively, without a lost marker. Thus a crash's
additional accounting loss is bounded by unsettled reservations: in-flight tool
runs and open per-holder tranches. The accounting cap cannot be enlarged by a
crash or forced takeover. A tool's **actual external spend** still depends on
correct estimates, reported usage, and its own provider caps. The optional
gateway on the pinned LiteLLM range provides only a soft cap, as described below. Reported overspend is recorded
and can exhaust the budget. Usage beyond the supported accounting range
terminally exhausts the whole run, even with unset caps.

## LLM gateway: hard and soft caps

The optional `llm_gateway` checks its strict profile before run ownership,
manifest publication, approval, or any virtual key creation. Approval warnings
include `LLM gateway uses a soft cap.` when soft-cap policy is accepted. **The pinned LiteLLM range
`>=1.107.0rc2,<1.108` cannot provide a hard cap.** Its profile is always
unverified. `require_hard_cap` defaults to true when `limits.max_usd` is set;
that configuration therefore fails preflight. Set it explicitly to false to
accept a soft cap:

```toml
[limits]
max_usd = 5.0

[llm_gateway]
url = "http://127.0.0.1:4000"
master_key_env = "LITELLM_MASTER_KEY"
require_hard_cap = false
```

Keep the master key in the named environment variable. Neither it nor a virtual
key belongs in configuration files, manifests or journal events. Gateway URLs
must have no userinfo, query or fragment. A run retains one adapter, its checked
profile and one key manager bound to the existing journal owner. Preflight
retains fixed method, route and status diagnostics from the built-in transport;
foreign adapter errors expose only their type. Gateway model identifiers must
be printable, nonblank and at most 256 characters, otherwise they display as
`<unknown>`. Errors and manifest evidence retain the first ten failures followed
by `and N more` when necessary.

A verified strict profile requires all of the following:

- Budget reservations enabled, with enforcement that fails closed when spend
  counters are stale.
- Known input/output prices and an output-token bound for every model, plus an
  enforced per-key bound on the request's `max_tokens`.
- Only routes that participate in budget reservations; pass-through endpoints
  disabled.
- An enforced maximum total request duration, no longer than the configured
  `request_timeout_s`.
- Virtual keys required for model access.

LiteLLM on the pinned range lacks three required guarantees: **request
`max_tokens` is not bounded per key**, **no maximum request duration is
enforced**, and **virtual keys are not required**. It also lacks the needed
settings introspection endpoint. The adapter checks liveliness and model
metadata, and reports `gateway settings introspection unavailable`; it never
assumes unreadable settings pass. `default_estimated_output_tokens` is an
estimate, not an enforced ceiling, and an HTTP timeout does not establish a
total request-duration guarantee. A batch-write interval is not proof that
spend logs are complete.

The manifest's `llm_gateway` block records the URL, `cap`, `verified`, `failures`
and `terminal_accounting`; it is identical in `manifest.json` and `run_started`.
Without a gateway the block is null. The pinned adapter records `cap = "soft"`,
`verified = false` and `terminal_accounting = false`. Candidate `trace` and the
board viewer display the key alias, reserved `max_budget` in USD and cap label.
They use the committed state captured with the board frontier, so a refresh
reveals later reconciliation while an already captured reader stays unchanged.

Gateway budgets cap **USD only**, not tokens. Pin the gateway's cost map to the
model prices used when reserving budgets; changing prices undermines estimates.
A soft cap does not guarantee that concurrent external requests stay within the
reservation. Keyed tool reservations stay fully charged for the run. With this
LiteLLM range, reconciliation keeps the full reservation labelled `estimated`:
without a verified duration bound, waiting longer cannot prove completeness.
Observed rows can still inform provenance without lowering the ledger charge.

## After the run: reconcile

Run `swarm-arena reconcile RUN_DIR`, where `RUN_DIR` is `<run_root>/<run-id>`.
It reads that run's `run.toml` by default; `--config PATH` selects an override.
The command prints each candidate, key alias, reserved USD, spend USD and label.
It checks storage conditional-write support (including an `s3://` run root)
before takeover, refuses a live owner, and holds the run lease while publishing
its report. Recorded gateway keys require the original gateway URL (userinfo
is ignored); a missing or changed binding is refused before takeover and
rechecked after acquisition. The driver also attempts reconciliation before its final commit,
using its existing owner and gateway profile. Gateway failures are recorded
without failing completion; storage failures and ownership loss still propagate.
A run with no committed head returns an empty report and releases its lease
without publishing history; it remains ineligible for resume.

The run directory and run root are trusted operator data. Anyone who can write
them can change the tools that resume executes and make resume or reconcile
send the environment master key named by `master_key_env` to the gateway URL in
that configuration. Keep run roots writable only by the operator. On shared
storage, pass `--config` from an independently trusted copy; the gateway
continuity check then refuses a changed URL. This does not authenticate a
modified journal or recorded plan.

Before reconciliation mutates a key, its journal and writer must match the
addressed run. The adapter checks the selected hash through `/key/info` and
compares the stored creation `metadata.run_id` and `key_alias` with the record.
A mismatch or unavailable identity leaves the key estimated without mutation,
using a fixed reason without external values. These checks use the pinned
LiteLLM key-info fields; they do not make an attacker-writable run root trusted.

A key's report becomes an observation only when all these conditions hold:

1. The executor confirms the original tool is dead. A key with no start or
   dispatch event counts as unused. A reused local process-group leader is dead
   only when both recorded and current start times are known and differ.
   A remote host or unavailable, stale or unbound Ray dashboard remains unknown.
2. The gateway confirms closure at time T. Repeated reconciliations preserve the
   earliest journaled confirmation; blocking alone is not confirmation.
3. The gateway enforces a maximum request duration no longer than
   `request_timeout_s`, reports a batch write interval, and both intervals have
   passed since T.

Otherwise the report keeps the full reservation as **estimated**, with the
failed conditions listed. The pinned LiteLLM adapter cannot verify the duration
and batch guarantees, so it stays estimated and does not delete keys for
completeness. An adapter that can demonstrate these capabilities reports
**observed (completeness unverified)** unless it also guarantees terminal
accounting, in which case it reports **observed**. Unverified observed spend
is not proof that every charge has arrived: late log batches may add rows.
Deletion is attempted only after a successful eligible read and durable recording.
The committed head must cover the observations and closure before deletion,
so a takeover after deletion can recover spend even after upstream log retention.

Reports deduplicate gateway requests by key and request identity, retain earlier
committed observations even when logs age out, and sum integer micro-dollars
before displaying USD. A key total that cannot be represented without losing a
micro-dollar stays estimated with `unsupported gateway observation` and is not
deleted. An unsupported run aggregate preserves all per-key results and reports
the exact full-reservation total as estimated with the same fixed reason. If
even that estimate cannot be represented, the aggregate is unavailable (`null`),
never rounded or clamped. These display limits do not prevent run completion or
its final review export. A later reconcile can add late rows for retained keys;
deleted keys reuse their committed observations without repeating gateway reads.
Readers retain their captured committed snapshot until refreshed.
Reconciliation never refunds reservations, changes a live run's budget, or
restarts the paid-work clock. Gateway waits have a separate bounded allowance;
healthy journal storage is not charged against that external-work timer. The
overall allowance is at most `settle_timeout_s`, capped before inherited
cancellation and writer closing/I/O deadlines. Each key has a fresh allowance
of at most `request_timeout_s` within that overall bound. An operation whose
allowance expired before entry reports `reconcile time allowance exhausted`,
without claiming a gateway failure. Reconciliation visits keys without confirmed
closure first. It checks identity once, immediately before blocking an unclosed
key or reading eligible spend; already-closed keys with no eligible spend need
no identity or process-death lookup. Repeated calls can therefore close later
keys even when the pinned profile never permits deletion. Deleted keys reuse
committed observations. Resume's closure loop also has one bounded overall allowance;
unclosed keys remain available for reconciliation. Run-end reads and
reconciliation finish before the final completion allowance is armed. Failure
events include the nonsecret key and candidate IDs for later recovery.
The KeyManager and reconciliation exception handlers do not convert
`KeyboardInterrupt`, `SystemExit`, `GeneratorExit`, or groups containing them
into gateway failures, or start compensating cleanup for those controls.
This statement applies to controls that reach these handlers. Caller cancellation
and registered ownership/terminal controls retain their existing handling.

A key created just before the open deadline but never journaled is unusable
(its bearer never left the driver), expires after `key_duration_s`, and is not
counted in reports.

## Tokens, USD, and tranches

Metered LLM calls reserve before dispatch against their scope's committed
tranche. Default tranche sizes are **0.25 USD and 25,000 tokens**. Small calls
use available tranche credit; a new tranche shrinks to the remaining capped
budget when necessary, and larger calls reserve independently. The advisory
hook checks tranche credit as well as the remaining ledger balance; the actual
reservation remains authoritative when calls overlap.

Each admitted call gets its own reservation ID and preallocated LLM call ID.
Callbacks settle by reservation ID, including concurrent identical requests.
The output cap never exceeds the catalog's `max_tokens`; estimates include
serialized prompts, input items, tools and output schemas. Provider-managed
retry and fallback controls, including policy overrides in extra arguments or
bodies, are stripped; numeric retries are forced to zero so hidden attempts
cannot share one reservation. Runner retries, when enabled, enter the metered
path again with a fresh reservation.

Scope owners drain before closing their tranche. A missing report after
`settle_timeout_s` produces an `estimated:` call error charged at the reservation;
a cancelled call is abandoned so drain does not wait for its callback. A
response-cache hit costs **0 USD and 0 charged tokens**, while retaining reported
usage for display. Local budget refusal before reservation sends no request,
creates no LLM call record and costs zero.

`BudgetExceeded` is both an SDK exception and a `LimitReached`. Registered
fencing, terminal cleanup failure and usage overflow stop the SDK operation
without another model turn; sanitized hook failures do not gain that authority.
The shared core classifier restores exact issued SDK carriers, including nested
carriers, from private provenance. Copied, caller-created or subclassed carriers
remain diagnostics. Caller cancellation takes precedence over carried controls.
Usage overflow retains conservative charges. Exact journal/ledger equality
applies to finalized, non-overflow accounting; fencing or interrupted publication
can leave a reservation conservatively charged. Gateway enforcement of external
tool spending remains later work.

## Three execution clocks and the watchdog

- **Tool clock:** an individual job uses the effective tool timeout bounded by
  the remaining round time and `Limits.tool_timeout_s`. A returned `Command`
  can tighten it further; its timeout and the job timeout take the smaller
  non-`None` value. Grace similarly takes the smaller Command/job value.
- **Round clock:** a round ends at the earlier of its own timeout and the
  remaining run time. Its managers and jobs are cancelled with grace and partial
  collection. Late results belong to the round that started them.
- **Run clock:** `run_timeout_s` counts active driver time accumulated across
  resumes. Downtime while stopped does not count. Expiry acts as `stop --now`.

The watchdog stops only the affected agent as `watchdog_stopped`, with a recorded
reason. It triggers after more than `max_refused_repeats` repeat refusals, after
`watchdog_idle_rounds` tool-running rounds without a new passing candidate, or
after `max_consecutive_failures`. `error`, `oom`, and `timeout` increment failures;
`ok` resets them. `provider_failed`, `lost`, `cancelled`, and `budget_exceeded`
neither increment nor reset that failure counter. Starved rounds do not count
as idle rounds. A resumed agent recovering its own identical committed request
does not incur a repeat refusal.

Mid-round core stops include run/goal budget exhaustion, deadlines, explicit
stops, and the watchdog. Turn, tool-run, and board-read caps end that agent's
round through `LimitReached`. Exceeding the learning publication cap is a soft
action error and does not itself end the round. Other policy pruning happens
between rounds. Tool exceptions
become error candidates; a manager exception ends only that agent's round.
User-raised cancellation and exception groups are contained at manager, action,
check, tool, and judge boundaries with type-only diagnostics. Judge construction,
batch preparation, anchor selection, and configuration validation use the same
classification before recording a failure. Healthy peers continue
and the goal can finish. Cancellation before dispatch refunds the tool reservation
exactly once, including command preparation and edit uploads. Cleanup has a bounded
settlement wait; work that may have started retains its charge. Actual caller
cancellation and ownership fencing still propagate. Fencing nested in an exception
group propagates as a bare ownership-loss error, without group or leaf values;
pending caller cancellation takes precedence. Hook messages, notes, and causes
are not rendered by the CLI. Type-only diagnostics retain the originating
exception type across sanitized copies; internal exception class names are hidden.
Request-time estimate and provider hooks run in registered, cancellable
preparation before reservation. Immediate stop and round deadlines abandon
these waits without using the executor collection allowance.
Startup estimates, provider declarations, executor hooks, and preflight failures
are also reported by type without hook values. A hook-raised cancellation is a
startup failure; cancellation requested by the caller still cancels the run.
Built-in provider probing and model-provider recovery are M3 work.

## Stop modes and resume

| Command | Current jobs | Finalization |
|---|---|---|
| `swarm-arena stop RUN_DIR` | Refuse new ordinary runs; let in-flight work finish | Judge, converge (including a converge team), and final judge from reserve |
| `swarm-arena stop --now RUN_DIR` | Cancel with grace and partial collection | Judge available candidates, select-only convergence; no converge team; retain best parts as `unjoined` |

Run budget exhaustion drains work and finalizes from reserve; goal budget
exhaustion affects that goal. Resume completes setup even when replayed usage
has exhausted the work budget, then finishes interrupted convergence and final
judging from the remaining reserve. Resuming a completed budget-stopped goal
reuses its recorded result without repeating paid work. Goals still queued at
any early ending never start and report `NO_RESULT` with reason `stopped`. A stop
arriving during convergence or final judging remains the lifecycle's termination reason.
Interrupted part joins retain their best parts as `unjoined`.

When budget exhaustion races with a new tool request, `LimitReached.limit` may
be `budget` at the budget gate or `stop` after the round closes admission.
Both outcomes refuse new work. Recovery retains the charged lost reservation
and the recovered ledger balances regardless of which refusal arrives first.

A graceful stop that preempts the round policy journals the default `Decision()`
without pruning. If a stop or deadline cuts a judging barrier short, candidates
that have no ranking score stay unranked even after resume. The round closes,
and later rounds judge their own outputs without rejudging those candidates.

After `stop --now` or the run deadline, final judging gets at most
`settle_timeout_s` (60 s by default). If cut short, selection uses available
committed ranking evidence and records judge errors; it cannot wait forever for
a fresh score. Executor shutdown is also shielded and bounded by that same
setting. A shutdown timeout preserves the original result or exception, and
shutdown failure never replaces an exception already raised by the run.

After a graceful `stop`, reserve-funded barrier and final judging, including a
paused-ranking retry at converge, retain their judge/round timeouts and the run
deadline. Their combined duration does not consume one `settle_timeout_s`
window. Each post-stop accounting/storage wait remains bounded, and real
drain/teardown has its own settlement allowance after that work. Teardown
deadline expiry raises the value-free `stop cleanup timed out` error.

Stop files are scoped to the live ownership epoch: `control/stop-e<epoch>` and
`control/stop-now-e<epoch>`, polled every two seconds. An old stop file does not
stop a resumed epoch. `Swarm.stop(now=True)` requests the immediate mode in
Python. CLI Ctrl-C prints `interrupted` and exits 130; it is an interruption,
not a promise that graceful finalization completed. Normal returned runs,
including no-result goals, exit 0; runtime/control failures exit 1 and usage
errors exit 2.

Resume from the committed journal with the saved configuration, or a human
replacement:

```sh
uv run swarm-arena status .swarm-arena/runs/<run-id>
uv run swarm-arena resume .swarm-arena/runs/<run-id>
uv run swarm-arena resume .swarm-arena/runs/<run-id> --config run.toml
```

`Swarm(config).resume(run_id, force=False)` provides the Python equivalent.
Resume trusts the saved configuration, journal and recorded tools. Restrict run
root writes to the operator and use a trusted `--config` on shared storage:
writers can redirect the configured `master_key_env` credential to another
gateway URL or change which tools resume runs. See the
[run-storage trust boundary](#after-the-run-reconcile) above.
Backend journal reads are contained and their bytes detached before chain or replay
validation. Backend diagnostics and backend cancellation cannot impersonate core
validation or caller cancellation.

An invalid or missing committed journal segment raises `ChainError`, preserving
the safe diagnostic that identifies the segment. Invalid replay payloads,
missing committed state, incompatible agent limits, and decreased budget caps
raise `ValueError` (or its `PlanError` subclass). These checks run before takeover.
Normally the previous lease must be released or expired. `--force` explicitly
permits a live takeover, records it, and fences the old owner by a conditional
ownership write; it does not refund already committed work. Resume restores
round counters, watchdogs, the recorded snapshot frontier and each agent's own
committed results. A committed terminal policy decision is finalized without
restarting managers. Missing result views can be rebuilt from the journal.

A goal stopped by `stop` or `stop --now` with rounds left continues even when
limits are unchanged. A goal stopped by a limit continues only when the resume
changes limits. Budget caps may increase or become unlimited, but cannot decrease;
changes are journaled as `limits_changed`.

A resumed stopped goal gets only its share of the remaining unallocated work
budget, which can be 0 when other goals were still running and already got its
unused budget; unless the resume raises the limits, such a goal stops again
with `goal budget` at its first run.

A continued goal's earlier lifecycle stays in the run journal: its `final_order`
and its `goal_finished` stop reason are kept there, even after the later lifecycle
replaces `solution.json` and the replay clears the goal's `end_reason`. `status`
and `solution.json` show only the latest lifecycle.

## Local isolation and secrets

**v1 sandboxing is not strong isolation.** Local jobs run as your OS user with
fresh workspaces, process groups, resource limits, bounded output, and an
environment allowlist. They can still access resources allowed to that user.
Judge-private files are withheld from tool staging, not protected against
arbitrary same-user code. Use trusted programs; containers are future work.
See the [isolation boundary](own-files.md#isolation-boundary).

`FunctionTool` returns JSON data only: it does not apply file edits or parent
file overlays. Like `ScriptTool`, its `env` lists variable **names**, which must
be in `RunConfig.secrets.allow`. These settings and their validation aliases
are human-only, not planner-settable or round-overridable. Keep resolved values
out of plans, requests, and journal events. Protect Python helper modules too:
editable `.py` files can cause import shadowing. Prefer data globs such as
`problem/*.physk`.

Local logs retain their last 8 MiB each; declared outputs default to 256 MiB
total and at most 4,096 files. Command preparation is abandonable: cancellation
releases its slot and ignores late results, but cannot forcibly stop Python
preparation code. Its workspace is retained with a warning for safe manual
cleanup. Collection runs once and has a 30 s bound. Output files and logs upload
one object at a time through the supervised storage boundary. Cancelling an
upload abandons its wait and prevents later uploads. Cancellation during the
process or collection also skips artifact uploads: the partial result keeps
collected data but drops file and log references. Process termination and
bounded collection still finish before data-only partial delivery. See the
[executor details](api/executor.md#localexecutor) before cleaning retained workspaces.

Direct `LocalExecutor` preparation preserves the exception object raised by a
constructor or command hook, except the documented `SystemExit` command failure
that produces an error output. Collection converts ordinary exceptions and
`SystemExit` to an error output with a type-only note, while cancellation,
interrupts, exception groups, and fencing-shaped exceptions propagate unchanged.
The engine classifies those transported outcomes at its supervised boundary,
where it can distinguish caller cancellation from a hook-raised failure.

A hook-created `Fenced` exception does not establish ownership loss, whether
raised directly or nested in an exception group. Only the journal's fenced
state or a core-registered fencing signal can terminate the run as fenced.
Caller cancellation takes precedence over even a registered fencing outcome.

Final judging records its complete order with the last score or judge error,
using the goal lifecycle, plan revision, judge reference, and exact candidate
membership as its batch identity. If a process exits after that batch and its
settlement commit but before final selection is published, resume reuses the
recorded order without another judge call or reservation. Error-only batches
also retain their no-result reason. A different lifecycle, revision, judge, or
candidate set requires a new final batch; ordinary barrier scores cannot stand
in for final completion. Recovery checks the recorded final inputs before
convergence ranks candidates or launches a team: final scores cannot reselect
the input batch. The completion key also binds the committed candidate set at
that boundary, so newly produced candidates invalidate reuse.

Synchronous library validation and import helpers propagate `KeyboardInterrupt`
unchanged. When the engine invokes these helpers in a daemon worker, the awaiting
engine boundary still contains hook-raised interrupts as type-only failures.
ScriptJudge reports the originating exception type for a single failed child;
multiple failures retain the exception-group type. Messages remain value-free.

Returned extension models are untrusted until preparation finishes. Score and
check results, executor outputs, plans, policy decisions, and manager reports
are detached and validated in abandonable daemon workers before the event loop
reads or serializes them. The prepared values use core model classes and copied
payloads, so subclass serializers and attribute hooks cannot block stop-now,
lease renewal, or fencing detection. The same rule applies to local executor
collection and conformance helpers. Learning records are constructed by core
code from the posting API and reconstructed by journal replay; there is no
extension-returned Learning hook. Detachment applies the core validation rules,
including UTC-millisecond timestamp normalization.

### Context identifier inputs

Public `AgentContext`, `JudgeContext` and `CheckContext` entry points copy
identifier and name arguments to exact strings using `str.__str__` before core
lookups, hashing, action admission or loop marshaling. This includes collection
elements (parents, evidence, refinements, tags and anchors), action and part
names, context run/batch/fidelity identifiers, and staging/edit paths.
`JudgeContext.judge_error(candidate_id, reason)` copies both strings and keeps
the first reason for each candidate. A string subclass therefore cannot supply
equality, hashing, conversion or formatting hooks to subsequent core operations.
Non-string identifiers raise `TypeError` naming only the parameter and required
type, without formatting the supplied value. Optional identifiers still accept
`None` where documented. For example, `ctx.judge_error("c1", "invalid result")`
records the same first-error diagnostic as before.


## Lease-owned storage

The driver awaits storage in abandonable daemon workers under its existing
stop, remaining run time and ownership supervision. Shutdown uses the configured
settlement allowance. Backend exceptions become value-free core errors;
trusted chain and replay validation keeps its own diagnostics. Returned blob
bytes and version identifiers are detached before the owner uses them.

Journal publication waits end at the last confirmed lease expiry.
Owner updates run against a private lease copy; only a timely result can update
the writer's live document and committed head. An abandoned storage request may
still finish remotely, but its return cannot trigger another driver operation.
Live edits and log files use a separate awaited boundary per object.
Synchronous published `BoardView` reads and owner-loop `Board.put_*` calls
retain their in-memory contracts.

A stop during the first journal publication settles that admitted batch within
the cleanup allowance before deciding whether the run started. If `run_started`
is committed, queued goals receive their stopped result and CLI summary; resume
skips those finished goals.

A run deadline closes queued goals as stopped, but it does not create a
`stop_requested` event. The status stop field records explicit `stop` and
`stop --now` commands only.

Startup-stop and driver-stop cleanup share the same bounded failure contract:
if their settlement allowance expires, they raise the value-free core error
`stop cleanup timed out`. Normal export failures retain their original public
exception shape while stop and deadline watchers remain active.
The same cleanup error applies when a journal storage wait exhausts settlement
while its confirmed lease remains valid. Confirmed lease expiry or takeover
still raises `Fenced`. Either failure disables subsequent publication; late
storage results cannot update the writer's state or start another operation.
After settlement exhaustion, a writer whose confirmed lease remains valid
attempts one bounded release so a new owner can resume immediately. A blocked
or unsuccessful release does not extend cleanup indefinitely or reopen publication.
This release attempt has its own `settle_timeout_s` allowance, capped by the
remaining confirmed lease. Closing the writer can therefore take up to about
2 × `settle_timeout_s`: one allowance for the original wait and another for release.

A driver stop or deadline also bounds an in-progress journal storage primitive,
including its retries, by the settlement allowance. Healthy graceful-stop work
may continue: the bound applies to each storage wait, rather than expiring the
writer after a fixed duration. Goal refresh and view storage use the same
settlement bound for graceful stop as for immediate stop and deadline.

A post-stop final-selection recovery read can exhaust its storage settlement
allowance. That failure propagates as a core storage-settlement error; it does
not publish judge errors or a terminal provider-failure result. Resume can
continue the unfinished goal from the committed journal.


## Provider failures

A provider outage pauses every registered agent in the run that uses that
provider, across goals. The provider set includes the manager's model catalog
entries and the tool's declared providers. An explicit tool
`failed_provider` identifies one provider; a provider-failed output without that
field applies to all providers declared by the tool.

For example, a tool can write
`{"data": {}, "failed_provider": "anthropic"}` to its result file. Agents using
Anthropic through either their manager or tool pause; unrelated agents continue.
The optional LLM adapter recognizes authentication, permission, rate-limit,
connection, timeout and service failures. Credit failures use structured error
fields first, with a message fallback confined to the adapter. Generic request-error
types do not suppress the credit-balance fallback; a specific identifying code
takes precedence. Provider messages
are never copied into engine errors or provider events.

Paused agents retain scheduler capacity. At the next fresh round, including
convergence, the engine probes each failed provider needed by that goal before
building its frontier or allocating shares. This includes newly queued goals
and converge teams even when no paused address remains for the provider.
Slot provider resolution runs concurrently, each bounded by the effective round
timeout (and remaining run time). A stuck constructor ends only that slot with
`runtime preparation cancelled`; known-provider probes and healthy peers continue. The probe uses the goal's judges
budget, a manager scope, input `ping`, and `max_tokens=1`; both response-cache
reads and writes are bypassed. Accounting drains before its tranche closes.
With no probe adapter or no catalog alias for the saved provider, recovery
succeeds without a model call. An ordinary probe error leaves the provider
paused and logs only the error type. A refused probe reservation ends that goal's
main rounds through the goal-budget stop path; other goals continue.

Recovery clears only the generation that was probed. A newer failure or another
unresolved provider keeps the agent paused. Recovery does not revive a pruned or
watchdog-stopped agent. An admitted SDK tool action finishes under core grace
before the manager drains; further actions are refused. For custom and scripted
managers, a provider stop closes tool admission and gracefully cancels running
tools so their partial results and settlements are collected before the agent
scope ends. The engine kill bound still limits unresponsive work. Caller cancellation,
lost ownership and terminal budget controls retain their precedence.

Provider generations and address causes are journaled and restored on resume,
independently of the current model catalog. A restarted partial round does not
probe or clear its saved causes, and still writes the paused agent's STOPPED
round status. A fresh resumed round may recover its reserved
slots without exceeding `max_agents_total`. Permanent disposition or goal
completion releases the reservation.


### Failures outside agents

Judges, panel members, replanners and curators use the same run-wide provider
latch as managers and tools. A provider outage in one goal therefore pauses all
matching agents, including later registrations, while their capacity remains
reserved until recovery or permanent disposition.

A failed non-ranking judge leaves its scores missing. A failed ranking judge
skips the round's policy and makes curation ineligible; it never authorizes
pruning on missing ranking evidence. At converge, paused ranking batches get
one retry through normal admission and the shared provider service. A distinct
batch identity makes a completed retry reusable on resume. Only a validated
provider failure on that retry can produce `provider_failed` when no finalist
passes; a recovered provider permits normal selection. A failed replanner keeps
the current revision and records a safe `plan_rejected` completion, including the original
`first_round` and any `join_retry`. A failed eligible curator records a completed
skip without a `curated` event. These receipts survive takeover, so a missing
decision or summary does not cause paid work to repeat.

Each settled invocation reports a failed provider once, even when several panel
members use it or a sibling is refused by the budget. A surviving panel carries
its `panel_drop` metadata to the core, which commits it with the survivor scores.
A drop describes a member loss, not whole-batch completion; single-member and
all-member failures are failed batches instead.

Failed judging and curation receipts are published only after accounting has a
known disposition. Storage failures, settlement timeouts, fencing, terminal
usage overflow and genuine caller cancellation are not provider failures and
cannot manufacture these receipts. Error messages carry identifiers and type
names, never prompts, outputs or provider diagnostic text.

A registered judge budget refusal at the barrier records `reserve_exhausted`
and ends ordinary rounds in select-only mode. At final judging it produces
`no_result` with reason `reserve_exhausted`. Registered limits in replanning and
policy retain their existing contained rejection/error-decision behavior.

A registered usage overflow from a manager or judge takes the run-budget stop path:
ordinary rounds end, and the free final selection can still run. A registered
cleanup or storage terminal failure escapes the run even when no ledger or
binding state accompanies it. Neither kind is recorded as a manager or judge error.
