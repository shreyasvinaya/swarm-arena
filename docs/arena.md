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
version 2; a version 1 cache must be rebuilt so old namespace routing is not reused.

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

`put_candidate`, `put_score`, `put_learning`, and `put_decision` retain local
hints after the core journals an item. They do not write the journal or publish
items to snapshots. Only board-level candidate, learning, and score lookups
can fall back to hints. Hidden learnings are excluded from normal reads,
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
A previously created view stays fixed. Storage reads and ingestion run under a
separate refresh lock. Board and view reads use prepared WAL reader connections
and plan projections, so an event-loop read does not wait for a worker fetching
a journal or writing the cache. The frontier and its prepared reader state are
published together under a short in-memory lock. A failed refresh leaves the
previous generation readable; snapshot preparation also belongs on a worker.
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
command's contract is to refuse while any process holds a cache lifetime lock;
`swarm-arena reindex` CLI integration is delivered separately. `close()` releases
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
host or endpoint. The planned `review import` command has the same requirement
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
Hiding a mistaken summary with the planned `learnings hide <summary-id>` command
undoes the merge's ranking effect; the original learnings remain intact.

Curation follows the committed nonterminal policy decision and precedes round
closure and replanning. No curator is called when configuration or new learnings
are absent, or after a terminal policy or stop. Constructors, model hooks,
invocation and result normalization participate in goal stop, deadline and
fencing supervision. Abandoned daemon work cannot allocate summary IDs or publish
results. Ordinary failures produce type-only diagnostics; genuine core budget
and fencing signals retain their control behavior. Pending unknown calls are
charged conservatively when the tranche closes.

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
