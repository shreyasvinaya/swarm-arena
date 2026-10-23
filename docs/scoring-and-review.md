# Scoring and review

Checks gate results, judges score them, policies prune between rounds, and a
final judging batch determines solution order. Script and trusted Python
judges run offline today. See the [scoring API](api/scoring.md),
[planning API](api/planning.md), [board API](api/board.md), and
[plan API](api/swarm_arena.md).

## Checks and manager feedback

Tool candidates are checked in plan order before publication to the manager;
cancelled preparations without a start skip checks.
`CheckResult.passed=False` makes a candidate **gated out**: it remains
recorded but cannot be judged or selected. A malformed result or check exception
fails closed. `ScriptCheck` also fails closed on crashes, timeouts, and missing
results. The core stamps each result with its check's plan name.

A check's `feedback` controls the manager's view:

- `"reasons"` (default) exposes its reasons and details.
- `"pass_fail"` exposes only the verdict; reasons and details are removed from
  manager-visible copies, including assigned convergence parents.

Checks and judges keep their private files out of tool workspaces. Feedback
filtering is not OS isolation; see [Limits and safety](limits-and-safety.md).
For a working check and scoring function, the integrated Rosenbrock source is:

```{literalinclude} ../examples/rosenbrock/rosen.py
:language: python
```

The [four-file script workflow](own-files.md#try-it-offline) also exercises a tool,
check, and judge without running managers. Use public
`swarm_arena.testing.read_output_file` and `read_logs` to read verified artifact
bytes from kit outputs; a run journal's relative log path is a different kind of
reference from the kit's content-addressed log reference.

## Scores, identities, and anchors

A larger `Score.key` wins, comparing tuple elements in order. `Score.minimize(x)`
uses `(-x,)`; `Score.maximize(x)` uses `(x,)`. `value` is a human-facing number,
not a replacement for the ranking key. Keys must be finite. Missing scores,
non-finite keys, or judge errors leave the affected candidate unranked; foreign
candidate IDs and later duplicate scores are discarded.

```python
from swarm_arena import Score

assert Score.minimize(0.25).key > Score.minimize(0.5).key
assert Score.maximize(2).value == 2
```

Each score is stamped with the judge's plan name, `rubric_hash`, and `panel_hash`.
Their tuple is `Score.identity`. Different identities are never compared within
ordinary ranking. Main goals and each part form separate ranking groups; ranks
also use one fidelity, by default the fidelity of the group's highest candidate
round. A new fidelity without a passing score does not revive older-fidelity
leaders. Ordinary ties break by earlier candidate round, then numeric trailing
ID counter, then full ID.

`ProgramJudge` calls a trusted scoring function. `ScriptJudge` executes a
scoring program with its own files. `Lexicographic` combines ordered judges:
only candidates scored by every member get a composite score.
With one judge the plan can omit `ranking`; with several, `ranking` must name
the declared judge whose scores drive selection. A part can inherit main
scoring or supply its own checks, judges, and ranking.

`Lexicographic` concatenates member keys in order, so each member must return
keys of one fixed length within a candidate batch. Different members may use
different lengths. If one member returns valid keys of differing lengths,
the whole batch receives judge errors and no composite scores are produced.
For example, a member returning `(1.0,)` for one candidate and `(1.0, 2.0)` for
another must pad or otherwise normalize its keys within the batch, even when
it is the only member of `Lexicographic`.

A judge can request `anchor_count()` earlier leaders. The core appends unique
leaders to new gated-in candidates and identifies them in `JudgeContext.anchors`,
so a judge can calibrate the batch against earlier evidence. Empty gated-in
batches skip judging. Anchors remain in the same group and comparison domain;
they do not authorize cross-part ranking.

## LLM judges

Install the `llm` extra and configure model aliases with the `judge` role.
`LLMJudge` scores each candidate independently against a fixed rubric and a
finite scale (0–10 by default). Include concrete score anchors in the rubric:

```python
from swarm_arena.llm import LLMJudge

judge = LLMJudge(
    "Correctness: 0 = unusable, 5 = partially correct, 10 = fully correct. "
    "Explain the score using evidence from the candidate.",
    ["opus", "gpt", "flash"],
    "median",
)
```

Aliases can refer to different providers. Every panel member scores every batch
concurrently, at temperature zero. `mean` (the default), `median`, `min` and
`max` aggregate votes. Repeated aliases retain separate votes and funding terms.
A member that fails any batch is excluded from the entire invocation; all
batches use the same surviving panel. A candidate is omitted unless every live
member gives it a finite value inside the configured scale. If all members
fail, no candidate receives a score. Budget refusals and core controls propagate;
they never remove a panel member.

The stable judge prompt includes the rubric and scale followed by the goal's
description and structured inputs inside an untrusted-content boundary. The
candidate batch follows in its own untrusted block. Changing the goal's
description or inputs changes the dispatched prompt and its response-cache key.

Ordinary batches contain at most eight candidates, including up to two earlier
leaders requested as anchors. Every batch re-scores the anchors, leaving room
for `batch_size - len(anchors)` other candidates. Configure these with
`LLMJudge.Config`; `anchors` must be smaller than `batch_size`. Each anchor gets
one score from the first batch, with later aggregated anchor values appended to
its notes so drift is visible. Its call links include every batch that held it.
The final judge scores all finalists in one call per member, regardless of
`batch_size`. Anchors and the final judge never read the response cache.

All four LLM judge aggregators preserve finite, in-scale votes as finite,
in-scale scores, including votes near the floating-point limits. Mean and
the median midpoint use exact-ratio summation before rounding to a float.

An empty or whitespace-only first-stage answer fails that member without
starting a formatting call. Its first-stage usage remains accounted.

After a panel member reaches a funding limit, the invocation admits no further
model calls. Already admitted calls finish and settle before the limit propagates.

Each LLM judge invocation drains its own child scope, so multiple LLM judges
can compose sequentially or recursively while their owner retains all call links.

A missing or out-of-scale anchor row in a later batch adds an `invalid` drift
note and retains the first score and the later batch's call links. An invalid
first appearance is not replaced by a later score.

The rubric identity includes the rubric, scale, aggregation and single-model
alias. A panel identity hashes its sorted aliases, retaining repeats; a reduced
panel records its surviving aliases, including a single survivor. Core stamping
and recursive `Lexicographic` composition preserve that identity. Aliases are
part of the identity: re-point a model under a **new alias** to avoid treating
its scores as comparable to earlier results.

Models without response-schema support use a free-text call followed by a
formatting call. The formatter is the cheapest eligible schema-capable alias
whose input, output and effective cache-write tariffs do not exceed the first
model's terms. Otherwise the same alias formats with the schema in its prompt.
The formatting output cap is no larger than either model's catalog cap. Each
call reserves through `MeteredModel`; there is no second barrier hold for an
LLM judge. Each returned score links the calls for its batch, including failed
member attempts. The judge drains its scope before returning, so every link
has a journaled call or an `estimated:` late-record placeholder.

The final slice holds the **output terms only**; inputs are reserved at call
time. A judge call whose serialized input (including schema) does not fit the
remaining budget is refused before any provider call, so the judge never
overspends through unfunded dispatch. `LLMJudge.judge` raises `BudgetExceeded`
on that refusal, including a formatting-call refusal. Already incurred spend
stays settled. Reported actual usage is never capped at the reservation.

## Pruning policies

Policies consume `RoundState` after a round barrier and return a `Decision` with
recorded reasons. Pruning compares only each group's current ranking identity
and newest fidelity. The four built-ins are:

| Policy | Exact retention rule |
|---|---|
| `KeepTopTeams(k)` | Keep the best `k` live teams in each group by team-best score. Unscored teams rank last; tied teams follow plan order. Prune teams below that cutoff. |
| `SuccessiveHalving(keep=0.5, min_per_team=1)` | Rank live, non-starved agents in each group by best result; keep `ceil(keep × agent_count)`, then additionally keep each team's best `min_per_team` agents with ranked results. Unscored agents follow scored agents in plan order. Prune other agents, or the whole team when all its live agents would be removed. |
| `Patience(rounds=2)` | After more than `rounds` history entries, compare the latest cumulative team best with the best `rounds` rounds earlier. Prune when there is no latest score, or when that earlier score exists and the latest key has not improved. Both use the current identity and fidelity. |
| `Chain(...)` | Apply member policies to the same state in order and merge prunes, keeping the first reason for each target. Guard the merged decision again. |

Every decision is guarded: already pruned, stopped, and watchdog-stopped agents
are excluded; a prune targeting a starved agent is dropped; the last live team
of a group is retained. This protection can keep more agents or teams than a
policy's nominal cutoff. A policy may also return `Decision.no_result` to end a
goal with a reason. Hard limits and watchdogs can stop work mid-round; they do
not wait for a policy barrier.

```python
from swarm_arena import Chain, KeepTopTeams, Patience, SuccessiveHalving

policy = Chain(KeepTopTeams(2), SuccessiveHalving(keep=0.5), Patience(rounds=2))
assert len(policy.config.policies) == 3
```

Completed barrier judging batches are recorded with their exact membership and
anchor IDs. After a crash, resume reuses committed scores and errors and pays only
for missing batches. A later round's anchor evaluation and the uncached final
judge remain separate evaluations.

## Convergence and the final judge

`ConvergeSpec(mode="select")` selects up to `top_n` eligible main-goal leaders.
Team mode runs a converge team, seeded with leaders or `parents_per_part` ranked
parents from each part. Assigned `AgentContext.seed_parents` are read-only and
are prepended to explicit tool parents, removing duplicate IDs. A manager's
`best(part=...)` reads its own group or a declared sibling group, including
`part=""` for the main group, and never an undeclared group. Reads automatically populate later candidates'
`read_ids` provenance.

Partial results require `converge.allow_partial=True`. `distinct=True` skips
repeated recorded fingerprints; an unknown fingerprint does not collapse
otherwise distinct candidates. Failed part joins can replan and retry only
within remaining rounds, budget, and `max_join_attempts`. Immediate stop or run
timeout suppresses the converge team and retains best parts as `unjoined`.

The ranking judge then re-scores all finalists together in **one uncached
batch**, with `JudgeContext.final=True`, `read_cache=False`, and no anchors.
The resulting keys determine final order; tied final keys use candidate ID
ascending. The final batch is journaled with `round=0`. Earlier scores remain
attached to candidates, so their first stored score is not necessarily the
final score. `top_n` and distinct selection apply to this final order.

Final judging uses the protected final slice. After an immediate stop or run
deadline it is bounded by `settle_timeout_s`; if interrupted, only available
committed ranking evidence can supply a fallback order and judge errors are
recorded. If final judging fails and supplies no final order, the goal records
`NO_RESULT` with reason `provider_failed`: this reason means the final judges
produced no order. Ordinary judge errors retain type-only diagnostics and do not
claim a provider outage. Only a validated provider signal creates a provider
transition or failed-provider receipt. A refused final reservation instead uses
reason `reserve_exhausted`. See [stop and resume](limits-and-safety.md#stop-modes-and-resume).

## No result and solution records

When no candidate survives checks and final judging, the goal records
`NO_RESULT` with a reason and writes `status="no_result"`; `Swarm.run` returns
an empty list for that goal rather than raising. A queued goal prevented from
starting reports reason `stopped`. A no-result solution's `reason` can differ
from the lifecycle's `goal_finished` termination reason, for example when a
stop leaves no passing candidate. `unjoined` also returns no main finalists;
its part selections are available in the solution file.

`goals/<goal-id>/solution.json` is a versioned view of committed journal state:

| Field | Contents |
|---|---|
| `schema_version`, `run_id`, `goal_id` | Stored format and run/goal identifiers |
| `status`, `reason` | `ok`, `no_result`, or `unjoined`, plus termination/no-result reason or null |
| `finalists` | Final order, with one-based `rank`, full `candidate`, recorded `scores`, direct-parent `lineage`, and zero-based final-evaluation `plan_revision` |
| `parts` | For `unjoined`, each part name and its candidate (possibly null), scores, and direct-parent lineage |
| `replicate_stats` | Cohorts by part, exact judge identity, fidelity, tool identity, and canonical inputs hash; `n`, `mean`, `min`, and `max` of finite score values when a cohort includes an explicit replicate |
| `best_of_n` | First finalist's ranking value and candidate ID when available; otherwise null |
| `plan_revisions` | Recorded plans with journal-authoritative origins |
| `manifest` | Run provenance metadata |
| `source_head` | Journal head from which this view was rendered |

`status` and solution views describe the latest goal lifecycle. Continuing a
stopped goal can replace its solution while earlier final orders and stop
reasons remain durable in the journal. See [Getting started](getting-started.md#read-the-solution)
for reading an actual solution.

## Human review

`HumanReview` exports committed candidates to `review.csv`, a static escaped
`index.html`, and one `candidates/<id>.json` file containing the candidate and
its recorded scores. Each goal contributes up to 20 rows by default. Main and
part groups use their own recorded ranking judges; replaced part identities
are excluded. Export reads one committed head and does not import judge plugins.

```python
from pathlib import Path
from swarm_arena import RunConfig
from swarm_arena.scoring import HumanReview

config = RunConfig.load(Path("runs/my-run/run.toml"))
review = HumanReview("alice")
out = await review.export("my-run", config=config, out_dir=Path("review"), top=20)
# After editing review/review.csv:
scores = await review.import_csv("my-run", out / "review.csv", config=config)
```

Reviewer names are 1–64 character lowercase slugs: `[a-z0-9][a-z0-9._-]*`.
Derived review object locations must fit the storage key and local path limits;
invalid locations are refused before takeover or receipt admission. Use `goal="goal-id"`
on export to select one main goal; `top` applies separately to each main goal
and includes its ranked parts.

The CSV columns are `id`, `goal`, `team`, `agent`, `scores`, `verdict`, `score`,
and `notes`. Keep the identifier and read-only context columns; edit the last
three columns. `scores` shows recorded judge/value pairs for reference. Export
prefixes text cells starting with `=`, `+`, `-`, `@`, a tab, or a carriage return
with an apostrophe so spreadsheets treat them as text. Canonical candidate/score
JSON remains unchanged. Context columns remain display-only during import. A
verdict is `accept`, `reject`, or empty, with any letter case. A score is a
finite number or empty. Both verdict and score empty means no review for that
row; notes alone do not create a score. UTF-8 BOMs, CRLF line endings, and quoted
multiline notes from spreadsheet programs are accepted. Missing columns,
unknown or duplicate IDs, bad verdicts and invalid/non-finite scores are
reported together with their row numbers before any storage probe or takeover.

`judge_name` is read-only and derives from the reviewer. The complete receipt is
validated before journal admission. Imports use judge `human:alice`, with key `(verdict_rank, score_or_zero)`:
accept is `1`, empty is `0`, and reject is `-1`. The numeric `value` remains
absent for a verdict-only review. Re-rank the arena using
`leaderboard --by human:alice` in the arena CLI.

Run imports **after the run ends**: a live owner prevents takeover. For `s3://`
roots, valid imports probe conditional writes before taking the lease. The CSV
is validated again against the final head fixed by takeover, and score IDs use
that new epoch. Import does not execute goals or start a run clock. The run's
active-time limit does not apply to later human review, including runs that ended
by that limit. A stop requested during import raises `RuntimeError("review import
stopped")`; lease and fencing supervision remain active.

Review publication and repair honor cancellation and fencing. An interrupted
publication resumes from the committed receipt; missing review objects are
recreated with their original names and bytes. Imports create arena objects only
for namespaces that receive scores. An empty import records no arena objects.


Retrying a CSV after a receipt committed creates a new
review batch; recovery itself does not duplicate the original batch.

Successful fresh and resumed runs also write an immutable bundle under
`<run_root>/<run-id>/review/e<epoch>-<seq>/` after the final commit. Rewriting
identical bytes is safe; conflicting bytes fail. Earlier completion bundles
remain available after resume.

Review imports publish to the configured `ArenaBoard` arena root when supplied;
otherwise they use `RunConfig.arena_root`.

Human judge leaderboards compare all fidelities together and order verdict/score
keys directly, without applying a program judge's `min_improvement` margin.


## Provider failures during judging

A validated provider failure produces no score or `judge_error` for the affected
judge batch. Healthy judges still run, independently for each main-goal or part
group. If a group's ranking judge fails, its candidates stay unjudged and the
round skips policy pruning and curation. The engine records that skipped phase,
closes the round, and publishes its page without inventing a policy decision.

An `LLMJudge` panel drops a failed member across the entire logical call, even
when the failure occurs in only one batch. Repeated aliases remain separate
votes and paid members. Survivor scores carry the hash of the sorted remaining
aliases, including when just one member survives; this is a new judge identity.
Nested `Lexicographic` judges retain those identities. All-member provider failure
propagates after accounting; a budget refusal is never a dropped panel member.

For example, a configured panel `["a", "b", "c"]` that loses `"b"` produces
scores under the identity of `["a", "c"]`, not its configured full-panel hash.
The failed provider also pauses matching agents in other goals through the shared
provider service.

Completed failed batches have durable receipts despite producing no scores or
judge errors. Resume reads those identities before preparing or funding another
batch. A failed final batch retains its original input membership, lifecycle,
revision and judge reference; resume completes `NO_RESULT(provider_failed)`
without selecting again from final scores or repeating the call. A completed
ranking-policy skip also prevents repeated manager or policy work.

Before initial converge selection, the engine retries each paused ranking batch
once through normal judge admission and the shared provider service. The judging
reserve funds this retry under a graceful stop or a run-budget stop; an immediate
stop or deadline skips it. Durable retry receipts and scores prevent repeat calls
on resume. Only a validated provider failure in this retry can make an empty
selection report `provider_failed`; otherwise it reports `no passing candidate`.
Explicit no-result outcomes from final judging retain their own reasons.
Batches paused in join-retry rounds are not retried and never determine the
`provider_failed` selection reason.
