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
Built-in model integrations and their model-cap checks arrive in M3; configuring
a model catalog alone does not provide them.

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
   and final judging. Curator integrations are future work.
2. Hold a protected **final slice for every goal**, including queued goals.
   It covers one convergence plus `max_join_attempts` additional attempts and
   the complete final ranking batch. Converge tools are estimated at **every
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
`settle_timeout_s` bounds the wait for pending call reports, not the subsequent
durable settlement writes. Without a committed settle, the reservation stays
charged in full. Cancelling construction or command preparation
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
correct estimates, reported usage, and its own provider caps; this release has no gateway
that enforces tool-internal provider spending. Reported overspend is recorded
and can exhaust the budget. Usage beyond the supported accounting range
terminally exhausts the whole run, even with unset caps.

The tranche machinery reserves before spending and settles unused portions;
its default sizes are listed above. Built-in LLM calls and gateway enforcement
are M3 work, not an available provider safety guarantee.

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
check, and tool boundaries with type-only diagnostics. Healthy peers continue
and the goal can finish. Cancellation before dispatch refunds the tool reservation
exactly once, including command preparation and edit uploads. Cleanup has a bounded
settlement wait; work that may have started retains its charge. Actual caller
cancellation and ownership fencing still propagate; hook messages, notes, and causes are not rendered by the CLI.
Request-time estimate and provider hooks run in registered, cancellable
preparation before reservation. Immediate stop and round deadlines abandon
these waits without using the executor collection allowance.
Built-in provider probing and model-provider recovery are M3 work.

## Stop modes and resume

| Command | Current jobs | Finalization |
|---|---|---|
| `swarm-arena stop RUN_DIR` | Refuse new ordinary runs; let in-flight work finish | Judge, converge (including a converge team), and final judge from reserve |
| `swarm-arena stop --now RUN_DIR` | Cancel with grace and partial collection | Judge available candidates, select-only convergence; no converge team; retain best parts as `unjoined` |

Run budget exhaustion drains work and finalizes from reserve; goal budget
exhaustion affects that goal. Goals still queued at any early ending never
start and report `NO_RESULT` with reason `stopped`. A stop arriving during
convergence or final judging remains the lifecycle's termination reason.
Interrupted part joins retain their best parts as `unjoined`.

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
cleanup. Collection runs once and has a 30 s bound. See the
[executor details](api/executor.md#localexecutor) before cleaning retained workspaces.
