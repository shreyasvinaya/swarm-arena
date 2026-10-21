# The arena

`ArenaBoard` reads committed results from runs that share a namespace. It uses
local SQLite caches and the existing run journals; it does not require a
database server. See the [board API](api/board.md) for the record types.

The engine uses `ArenaBoard` by default when `RunConfig.board` is `None`.
Fresh and resumed runs publish their namespace pointers after taking the lease,
then read other runs through their recorded round-start frontier. Selection and
budget decisions use only the current run's state.

A repeated request from another run still executes. The returned candidate's
warnings include `past-run match <id>: <judge>=<value>`, or `unscored` when no
ranking score exists. The hint uses the newest complete counted match visible
at round start; a same-run repeat still follows the normal refusal or reuse
rules. Managers may select visible past-run candidates as parents.

`ctx.best(k)` reads the agent's own group. `ctx.best(k, part="name")` selects a
declared sibling with that group's ranking judge, and `part=""` selects the
main group. These reads keep the same frontier, charge one board read, and
record the returned IDs. They do not widen candidate-file or feedback access.
The offline `check_manager` harness uses a private temporary arena and cache.

A part may use a different namespace from its main goal. Its candidates, scores,
and learnings are indexed in that part's namespace; declared sibling views keep
the same recorded frontier while selecting the sibling's namespace. A main view
can still read its declared parts and join their results. The cache schema is
version 4; incompatible older caches, including versions 1, 2 and 3, must be
rebuilt with `swarm-arena reindex`.

## What the arena stores

Run journals are the record. The shared arena contains pointers and human
review and curation objects:

```text
run_root/<run-id>/
  owner.json
  journal/e<epoch>-<seq>.jsonl
arena_root/<namespace>/
  runs/<run-id>.json
  reviews/<reviewer>/<timestamp>-<8-hex>.jsonl
  curation/<timestamp>-<8-hex>.jsonl
```

Pointers, journal segments, reviews, and curation objects are write-once. The
run's mutable `owner.json` names its committed journal head. Only the chain
reachable from that head counts; orphan segments never enter a view. Empty
review and curation objects are valid and remain part of recorded frontiers.

`put_candidate`, `put_score`, and `put_learning` retain local
hints after the core journals an item. They do not write the journal or publish
items to snapshots. Only board-level candidate, learning, and score lookups
can fall back to hints. `put_decision` accepts a notification without retaining
a second copy of the journaled decision. Hidden learnings are excluded from normal reads,
including those fallbacks.

## Snapshots and frontiers

`frontier()` makes one authority read per run and records its exact committed
head along with the names of the included human objects. Broken neighboring
runs produce warnings and are skipped. An owning run's read errors propagate;
an owning run with no commit yet simply contributes no mark.

The same frontier gives the same view when reopened with the same goal and
owning-run context, including on a machine with a fresh cache. Later commits,
reviews, and curation cannot change its membership or search corpus. Deleting
a pointer removes the run from later snapshots only: an existing frontier
retains its recorded root and head. Those recorded roots must remain reachable
when rebuilding a missing cache.

A run resumed from another local path is read through its new path by its own
driver. Pointers are write-once, so other readers keep its first pointer and
see the run only while that path is reachable. Keep a shared local run_root at
one stable path.

```python
from swarm_arena import Goal
from swarm_arena.board import ArenaBoard

board = ArenaBoard(arena_root="./arena")
board.open("", "", ["toy"])  # attach a reader without publishing a pointer
try:
    frontier = board.frontier()
    key = Goal(id="optimization", namespace="toy").key()
    view = board.snapshot(key, frontier)
    candidates = view.best(key, 5, judge="quality")
    hints = view.search_learnings("toy", "convergence")
    visible_count = view.count_learnings("toy")
finally:
    board.close()
```

Main views select part identities from the owning run's applicable committed
revision at the frontier. A changed input or file manifest creates a new part
identity: old evidence stays readable by its explicit historical key, but does
not enter the current part's rankings or parent membership. Removed parts are
not current siblings. A reader without an owning run uses the first matching
run and goal in identifier order. Declared sibling capabilities retain the
same frontier and revision; ordinary part-view reads retain their own scope.
Equal part keys can share candidates across runs even when their part names
differ. Candidate records and leaderboard rows retain their original provenance.
When the opening run has a main or part group with the requested key, that
owning group takes precedence over matches from other runs, including a foreign
main goal with the same key as an owning part. Its main and producing keys
determine the candidate pool; its plan determines siblings and learning scope.
Direct part snapshots and sibling capabilities agree on these scopes. With no
owning match, main-goal matches still precede part-goal matches.

Rankings use the owning run's recorded judge identity. A reader without an own
run, or a run with no score for that judge yet, uses the newest recorded identity
in the selected group. A caller's `run_id` filter limits candidates, independently
of that identity choice. Hidden learnings never appear in lists, searches, or
counts. Part-scoped learning reads use the resolving main goal; namespace-wide
reads still prefer that goal after higher-priority relevance criteria.

Call `frontier()` on a worker to publish committed data before board-level reads,
and again to refresh it. Before the first publication, committed reads are empty;
candidate, learning and score ID lookups can still use process-local hints.
A previously created view stays fixed. Refreshes leave the published data
readable, and a failed refresh preserves the previous view. Snapshot preparation
also belongs on a worker. Close the board after use; reads after close raise
`RuntimeError`. Reopening starts a new lifetime, while old views remain closed.
Plan resolution is retained per run and committed mark in the open board.
Repeated reads at unchanged marks do not replay journal history. Advancement
reuses committed replay checkpoints and folds only missing events; a cold or
historical prefix may require one reconstruction. Siblings reuse the fixed
resolved plan, so later ingestion cannot change their authority.
Lineage holds one shared frontier read scope per namespace and queries ancestor
IDs in breadth-first batches. Each requested ancestor is looked up at most once,
including missing parents and cycles; temporary sets are released on exit.

## The digest

At round start, `ctx.digest` describes the fixed snapshot. It includes up to
five leaderboard rows per group, with candidate IDs, team/agent, score, and
parent IDs. Each header names the ranking judge and the first eight characters
of its rubric and panel hashes. Main views include a block for each scored part
under that part's own judge; part views show only their own leaderboard.
A candidate's latest visible human accept/reject verdict appears on its row.
Group membership follows the fixed GoalKey capability, including candidates
recorded under another run's different part name. Their recorded provenance is preserved.

Pinned learning titles come first, followed by up to ten other titles in the
view's relevance order. These lists cover the whole namespace, including other
goals and past runs, and exclude hidden learnings. They include IDs and verified
markers on top learnings, so agents can use `read_learning` for bodies and evidence.
Board rows and titles sit between `<<from other agents, unverified>>` and
`<<end of content from other agents>>`.

The final line counts every visible learning not listed, gives the visible
count for this goal, and directs agents to `search_learnings`. From round two
onward it also counts the owning run's visible learnings from the previous
round, across all goals in this namespace.

`limits.digest_tokens` defaults to 1,500. The cap counts strict UTF-8 bytes, an
upper bound on tokens, over the entire returned string, including markers,
headers, and counts. More titles need a higher `digest_tokens`. To fit, the
digest removes top titles from the end, then pinned titles, then part rows,
then main rows. Empty part blocks disappear. With very small caps only the
markers and counts remain; if those do not fit, the counts line is clipped at
a UTF-8 character boundary. Agents search on demand for more context.

## Posting learnings

Use `await ctx.post_learning(title, text, evidence, ...)` to share a result.
Titles must be nonempty single-line text of at most 120 characters after trimming;
control characters and Unicode line separators are refused. The Markdown body
can contain at most 1,500 characters. Title and body validation errors include
the posting guide, `LEARNING_GUIDE`:

> Write for a colleague: what you tried, what happened, and the number.

Every agent post needs candidate evidence from its fixed round-start view or
its own items from this round. Other agents' new items become available after
the round barrier. Human authors (`human:`) may post without evidence.
`verified` is set by the core when at least one supporting candidate is complete,
has status `ok`, and has no failed checks. Partial, failed, or gated-out evidence
can support an observation without verifying it.

The duplicate check compares lowercase title word tokens with the top bm25
search hit in the same view and with the agent's own visible learnings. A Jaccard
similarity of at least 0.6 requires either `refines` naming a detected duplicate
or a nonblank `duplicate_reason`. An unrelated refinement does not satisfy the
rule. Hidden learnings do not participate. Duplicate refusals identify learning
IDs without echoing titles or bodies.

```python
learning = await ctx.post_learning(
    title="Fitting log(nu) converged in 8 iterations",
    text="The raw-nu fit took 24 iterations on the same grid.",
    evidence=[candidate.id],
    refines=[previous_learning.id],
)
```

Posting charges one action turn. Invalid posts allocate no learning ID and do
not consume the learning quota. The `max_learnings_per_agent_round` cap remains
a soft error, allowing other actions to continue. `ScriptedManager` supplies
`duplicate_reason="scripted manager: one learning per round"` when posting its
best visible scored evidence each round.

## The local cache

Pass `cache_dir` to choose the directory. Otherwise the board uses
`$XDG_CACHE_HOME/swarm-arena/<hash>`; when `XDG_CACHE_HOME` is unset, its base is
`~/.cache`. The hash is the first 12 hexadecimal characters of SHA-256 over the
arena root, with local roots resolved to absolute paths before hashing. Each
namespace has its own `<namespace>.sqlite` file, opened only on first use.

The cache is disposable and can be rebuilt from reachable recorded storage.
It is safe to delete only when no swarm-arena process uses it. The reindex
command refuses while any process holds a cache lifetime lock. Run
`uv run swarm-arena reindex --config "$ARENA_CONFIG"` after the command-line
example below to rebuild its caches. `close()` releases
all cache connections and their locks. Board state and cache operations are
synchronized across worker and caller threads.

Configured boards use the same lifecycle boundary on run and resume. Construction,
`open()`, and `close()` report only exception types for connector failures;
registered core signals retain their vetted meaning. Opening can be cancelled.
Closing is shielded for at most `RunConfig.settle_timeout_s`, preserves a primary
run failure, and abandons a blocking connector call when that deadline expires.
The synchronous call runs in a daemon thread and may continue after abandonment.

## Storage on S3

Each of `run_root`, `arena_root`, and `artifact_root` can be a local path or an
`s3://bucket/prefix` root. Install S3 support with:

```console
uv add 'swarm-arena[s3]'
```

For example, set the roots in your run configuration:

```toml
run_root = "s3://my-bucket/runs"
arena_root = "s3://my-bucket/arena"
artifact_root = "s3://my-bucket/artifacts"
```

Credentials and the endpoint come from the standard AWS environment, including
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and
`AWS_ENDPOINT_URL_S3`. Do not put them in `run.toml`. A local `run_root` with an
S3 `arena_root` is refused because other readers need access to the run journal.

The endpoint must enforce both `If-None-Match` creation and `If-Match`
replacement. Preflight verifies these conditions once per distinct S3 root.
`resume` probes again before taking over a run, since it may run on a different
host or endpoint. The `review import` command has the same requirement
before takeover. `S3Store` also refuses each write if the installed botocore model
would discard its required conditional parameter. Listings bypass s3fs caches
so newly published run pointers are visible on the next listing.

Each probe writes one small object under `.swarm-arena-probe/` within each S3
root. The objects are retained because `Store` has no delete operation. Configure
a bucket lifecycle rule for each root's probe prefix to expire old probes.
Local stores require no probe objects.

Conditional PUTs use a dedicated client with automatic retries disabled in
s3fs, the SDK, and the underlying HTTP transport. A lost response or transport failure raises
`WriteOutcomeUnknown`, even if the server committed the write. Reads and listings
retain their retry settings. Only an observed HTTP 409 conflict uses the bounded
Store retry policy; callers must reconcile an unknown write outcome by reading
stored state.

Child-store prefixes follow the raw Store key rules: characters such as `?`,
`#`, `%`, and spaces remain part of the key when opening nested child stores.

Listings match the local Store contract: basenames containing `.tmp-` or ending
in `.lock` are hidden, including nested objects. Filtering applies to every page
before returning sorted keys relative to the store root.

## Curation

Set `RunConfig.curator` to a reference to a `swarm_arena.board.Curator` subclass
when a run should merge related learnings at ordinary round end:

```python
from collections.abc import Sequence
from swarm_arena import Ref, RunConfig
from swarm_arena.board import BoardView, Curator, Learning


class MergeRelated(Curator):
    async def curate(
        self, namespace: str, new: Sequence[Learning], view: BoardView
    ) -> list[Learning]:
        if len(new) < 2:
            return []
        return [
            Learning(
                id="proposal",
                namespace=namespace,
                run_id="",
                goal_id="",
                round=0,
                author="human:proposal",
                title="Related observations",
                text="These observations share the cited evidence.",
                supersedes=[item.id for item in new],
            )
        ]


# Put the class in an importable module, for example my_project.curators.
config = RunConfig(curator=Ref(path="my_project.curators:MergeRelated"))
```

Curators follow the normal component configuration contract: an empty safe
configuration by default, or a nested `Config` model. `models()` declares model
aliases and defaults to `()`. The hook runs inside the curator LLM scope, with
holder `<goal>/:curator` paid from the reserve. This hook adds no built-in LLM
curator; `LLMCurator` is planned for the LLM milestone.

Proposals must supersede at least one learning from `new` or the visible
namespace snapshot. Hidden and unknown learnings are refused. The core validates
content with the production `Learning` rules, assigns IDs and run/goal/round
metadata, sets `author="curator"`, clears pin/hide and requesting-call metadata,
and combines evidence in supersedes order without duplicates. A summary is
verified when any superseded learning is verified. Evidence-free summaries are
dropped, including summaries of human learnings that have no evidence.

Superseded learnings remain searchable but rank below their visible summaries
in the normal learning list. Search still prioritizes text-match relevance.
Hiding a mistaken summary with the `learnings hide <summary-id>` command
undoes the merge's ranking effect; the original learnings remain intact.

Curation follows the committed nonterminal policy decision and precedes round
closure and replanning. No curator is called when configuration or new learnings
are absent, or after a terminal policy or stop. Constructors, model hooks,
invocation and result normalization participate in goal stop, deadline and
fencing supervision. Abandoned daemon work cannot allocate summary IDs or publish
results. Ordinary failures produce type-only diagnostics; genuine core budget
and fencing signals retain their control behavior. Pending unknown calls are
charged conservatively when the tranche closes.

Cancellation-shaped failures raised by a curator hook also produce a type-only
error receipt and allow the goal to continue. Real caller cancellation retains
its cancellation behavior.

Synchronous curator hooks run in abandonable daemon threads: the engine abandons
their waits at stop-now and on timeouts. Async curator hooks run on the engine
loop and must cooperate with cancellation. They must not block the loop or
shield their waits; either can delay stop-now completion and is outside the
abandonment guarantee. Use a synchronous hook when its body needs abandonment.

A curator reserve refusal ends ordinary rounds with `reserve_exhausted` and
selects from existing candidates. It does not schedule a replan, even when the
refused request leaves funds available in the ledger.
Its `reserve_exhausted` control event carries `reason="reserve_exhausted"`
before the completion receipt is admitted. Replay restores that terminal reason
even when the process dies before the receipt or round closure. Recovery closes
the interrupted round without replanning or another curator/ordinary work call.

A successful preparation publishes all summary `learning` events and one
`curated` receipt together, without an await between their appends. Its plain
v1 payload contains `summaries` (IDs), `supersedes` (summary ID to source IDs),
`errors` (value-free diagnostics), `round`, `revision` (the executed revision),
and `decision_seq` (the committed policy event sequence). The event's goal ID
and the policy sequence distinguish phases across revisions and lifecycle
continuations. Engine events also retain the shared run-clock checkpoint.
Empty results and ordinary errors complete an attempted phase; disabled and
no-new-learning skips do not emit receipts. Stop-abandoned attempts cannot
publish summaries. Replay indexes summaries through their learning events and
uses the receipt only to prevent repeated curation. Recovery without a receipt
may retry an eligible nonterminal attempt; terminal recovery performs no new
paid curation. Curation does not change join-retry or pending-replan state.


## Round pages

Each closed ordinary round creates a small Markdown page at
`<run_root>/<run-id>/goals/<goal-id>/rounds/round-NN.md`. The page is published
only after the round-close commit, including committed curator summaries, and
before the next live replan. It never grows afterwards: later rounds, reviews,
curation objects, revised plans, and changed limits cannot change its contents.

For example, the body of a page can look like this (the stored file also starts
with a source-head comment):

```markdown
# physika / ns-viscosity · run example · round 2 of 3
Leaderboard (rel-error, lower is better)
| # | Candidate | Team / agent | Score | Built on |
|---|-----------|--------------|-------|----------|
| 1 | c-0193 | fd-solver / tuner-1 | 0.0041 | c-0112 |
| 2 | c-0171 | spectral / author-b | 0.009 | — |

New learnings (1)
- L-0088 **Fit log(nu)** fd-solver/tuner-2 · verified
  Converged with fewer steps. Evidence: c-0193 (0.0041)

Pruned: team `surrogate` (no improvement).
Full board: `swarm-arena board serve example` · 412 learnings in namespace `physika`.
```

The leaderboard shows at most ten rows, with each part ranked under its own
judge and producing goal identity. Every displayed field uses the committed
prefix through that round's closure and the revision that executed the round.
The namespace count uses the recorded round-start frontier plus that goal's new
committed learnings, including curator summaries. Evidence outside this run's
recorded history is labeled `unavailable` rather than looked up in a newer view.

Resume repairs missing pages for every committed closure, including finished
goals, terminal policies, and join retries, without new manager, judge, or
curator work. Snapshot and count hooks run in abandonable daemon threads under
goal supervision on both live close and resume. Cancellation releases the owner
without waiting for a blocked hook; a late result cannot publish a page.
Hook failures retain registered core controls; other failures expose only their
exception type, without hook messages, notes, or causes.
Publication is create-only: any existing page remains untouched, including one
created by a peer after the initial existence check, regardless of its source
head or contents. Other view types retain their conditional replacement policy.
The segment containing the original close supplies the page's source head.
Round pages are derived views; the committed journal
and the recorded frontier remain their source of truth.

## Command line

Every arena command accepts `--config PATH` (default `run.toml`) and works without
connecting to a running driver. Configure an `ArenaBoard`, or leave `board`
unset to read `arena_root` with the default local cache. Each invocation refreshes
its committed frontier before reading. Run selectors use IDs such as
`2026-10-21T00-00-00Z_123abc`; the existing `status`, `stop`, and `resume`
commands continue to take a local `RUN_DIR`.

| Command | Behavior |
| --- | --- |
| `review export RUN [--goal ID] [--top 20] [--out DIR]` | Export CSV, candidate JSON and HTML; default directory `./review-RUN/`. Print its path. |
| `review import RUN CSV [--reviewer NAME]` | Validate all rows before taking the ended run's lease; print `imported N scores as human:NAME`. Reviewers use lowercase slugs of 1–64 characters; the login name is the default only when it is a valid slug, otherwise supply `--reviewer`. Export needs no reviewer identity. |
| `leaderboard RUN_OR_NAMESPACE [--goal ID] [--by JUDGE]` | Rank each current main/part identity with its own recorded ranking judge, or use the named judge for every group. A run limits rows and uses its improvement threshold; a namespace includes all runs under the newest score identity. |
| `board search QUERY (--namespace NS \| --run ID) [--goal ID] [--tag TAG] [--verified] [--human] [--pinned] [--hidden] [--limit 10]` | Print matching learning IDs, titles, authors and markers. Tags repeat. `--hidden` selects hidden entries; ordinary searches exclude them. An empty query lists by relevance. |
| `board show ID (--namespace NS \| --run ID)` | Show a candidate's request, status, checks, scores, parents and changed filenames, or a learning's full text and evidence scores/errors. |
| `learnings pin ID --namespace NS [--author NAME]` | Create one immutable curation object. Unknown namespace-local IDs fail without a write. |
| `learnings hide ID --namespace NS [--author NAME]` | Create one immutable hide object. Hidden targets remain available to explicit curation actions. |
| `learnings add --namespace NS --title TITLE --text TEXT [--goal ID] [--tag TAG] [--evidence ID ...] [--author NAME]` | Create one human learning, print its `L-h-` ID, and verify evidence against the refreshed frontier in that namespace. Author defaults to the login name. |
| `trace CANDIDATE (--namespace NS \| --run ID)` | Print committed plan provenance, run details, checks, scores and learning citations. |
| `reindex [--namespace NS ...]` | Rebuild selected disposable caches, print `NS: N runs`, and report cache warnings on stderr. Without namespaces, rebuild existing local caches. |

Leaderboard headers show `namespace/goal[/part]`, judge, and eight-character
rubric/panel hashes. Tab-separated rows contain rank, candidate ID, run ID,
`team/agent`, value, and parent IDs (or `-`). Verdict-only human reviews display
`-` for value and still rank by verdict. With no scores, the command prints
`no scores for JUDGE`. Board reads and reindex report cache ID-clash warnings
on stderr. Exit codes are 0 for success, 1 for runtime errors, 2 for usage,
and 130 for interruption.

From the source checkout root, after `uv sync --all-extras --all-groups`, this
complete example runs the included Rosenbrock plan, captures the run ID from its
output, and captures a newly added learning ID before pinning it:

```sh
ARENA_CONFIG=examples/rosenbrock/run.toml
ARENA_RUN_OUTPUT=$(uv run swarm-arena run --yes "$ARENA_CONFIG" examples/rosenbrock/plan.toml)
printf '%s\n' "$ARENA_RUN_OUTPUT"
# The CLI's "run ID at DIRECTORY" line supplies the generated run ID.
ARENA_RUN_ID=$(printf '%s\n' "$ARENA_RUN_OUTPUT" | awk '$1 == "run" {print $2}')
uv run swarm-arena leaderboard "$ARENA_RUN_ID" --config "$ARENA_CONFIG"
ARENA_LEARNING_ID=$(uv run swarm-arena learnings add --namespace toy --title "Rosenbrock baseline" --text "Compare gradient descent and random search." --author alice --config "$ARENA_CONFIG")
uv run swarm-arena learnings pin "$ARENA_LEARNING_ID" --namespace toy --author alice --config "$ARENA_CONFIG"
uv run swarm-arena board search "Rosenbrock baseline" --namespace toy --human --pinned --config "$ARENA_CONFIG"
uv run swarm-arena review export "$ARENA_RUN_ID" --out ".swarm-arena/review-$ARENA_RUN_ID" --config "$ARENA_CONFIG"
# For this demonstration, accept every exported candidate with score 1.
# For a real review, edit those verdict/score cells in a spreadsheet instead.
uv run python - "$ARENA_RUN_ID" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(".swarm-arena") / f"review-{sys.argv[1]}" / "review.csv"
with path.open(newline="") as stream:
    reader = csv.DictReader(stream)
    columns, rows = reader.fieldnames, list(reader)
for row in rows:
    row["verdict"], row["score"] = "accept", "1"
with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
PY
uv run swarm-arena review import "$ARENA_RUN_ID" ".swarm-arena/review-$ARENA_RUN_ID/review.csv" --reviewer alice --config "$ARENA_CONFIG"
uv run swarm-arena leaderboard "$ARENA_RUN_ID" --by human:alice --config "$ARENA_CONFIG"
```

## Board viewer

From the example setup above, serve its namespace locally:

```bash
uv run swarm-arena board serve toy --config "$ARENA_CONFIG" --author alice
```

You can also pass a recorded run ID in place of `toy`; the index then
filters to that run by default. `--host` defaults to `127.0.0.1`, and `--port`
to `8765`. Use `--port 0` for an available port; the flushed `Serving http://…/`
line reports the actual port. Ctrl-C stops this command successfully (exit 0).

The viewer binds locally by default. Explicitly setting a non-loopback `--host`
exposes it on that interface. Every request must have a `Host` matching
`127.0.0.1:<bound-port>`, `localhost:<bound-port>`, or the configured host and
bound port. This check prevents DNS rebinding from exposing the form token.
The pages use escaped HTML with no JavaScript or external resources.

The index has 50 learnings per page and preserves filters in Previous and Next
links. Query parameters are `q` (literal text search), `ns`, `goal`, `run`,
`part`, `team`, `agent`, `round`, and `tag`; `verified=1`, `human=1`, and
`pinned=1` select those flags. Hidden learnings are excluded by default:
`hidden=1` selects only hidden items, while `hidden=all` includes both. For
example, `/?q=viscosity&verified=1&page=2` shows the second page of verified
matches. Each request uses one published frontier; incoming requests refresh
it in a background worker at most once every five seconds. Once a generation
exists, GET requests keep reading it during storage refresh or mutation I/O.

A learning page shows its text, author, markers, and linked evidence with
formatted scores or an unknown-candidate error. Candidate pages show a summary,
linked parents and ancestors, and the committed provenance trace. Hidden
learning URLs remain available for human curation. Pin, hide, and add forms
require the per-server token and each write one immutable curation object,
using the same validation and mutation operations as the terminal commands.
Add accepts a configured namespace, title, text, optional goal and comma-separated
tags. New human learnings have an `L-h-…` identifier and a `human:<author>`
author. Each successful mutation refreshes the published frontier before its
303 redirect, so its learning page and updated markers are immediately readable.
A POST may wait for an in-progress storage refresh; GET requests continue reading
the published generation during that wait.

Foreign, corrupt, unreadable, or newer-version review and curation objects are
skipped with a warning when capturing a new frontier. Recorded snapshots still
require every object named by their frontier.

When candidate IDs collide, journal scores for a discarded producer do not
rank the retained candidate. Scores recorded for external visible candidates
and named human reviews remain available. External scores stay with the producer
selected by their recorded scoring frontier; ambiguous references without a
recorded frontier are omitted. Historical frontiers resolve their own retained
producer.

Large arenas still retain committed event/replay data in memory, and fixed views
materialize candidate models. Refresh I/O can repeat across namespace caches.
Memory and I/O therefore grow with arena history; reducing these costs is planned
for M3/M4.

A run ID supplied to `leaderboard` or `board serve` is resolved from stored runs
or arena pointers before treating the target as a namespace. Custom IDs such as
`my-run` are supported. In the viewer, clearing the Run field includes other runs
and human learnings; blank optional filter fields impose no restriction.

Board snapshots, curation reads, and round-page storage run in abandonable workers.
Cancellation, immediate stop, and ownership loss can release the run while a
backend call is still blocked. A cancelled page probe cannot start a later write.
Page storage failures report only the originating exception type, including
during resume repair; backend messages, notes, and causes are not exposed.

When a revision introduces a namespace, the running board attaches it before
agent dispatch. Live and resumed boards retain the union of all admitted revision
namespaces. Attaching a namespace preserves reads from the published frontier.

Resuming an unfinished round restores each agent's own committed learnings for
reads, search, duplicate checks, and refinement. Peers still read the original
round frontier, so another agent's same-round learnings stay private until closure.

Storage preflight and resume receipt repair use the configured `ArenaBoard`
arena root, matching review import/export and board reads. The top-level
`arena_root` is the fallback when the board does not supply an arena.

Round-page rendering returns the Markdown body. Publication adds exactly one
source-head comment and checks only the newly closed round; resume still repairs
all missing closed-round pages. Existing page bytes remain untouched.
