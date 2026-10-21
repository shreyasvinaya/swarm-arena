# The arena

`ArenaBoard` reads committed results from runs that share a namespace. It uses
local SQLite caches and the existing run journals; it does not require a
database server. See the [board API](api/board.md) for the record types.

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

Rankings use the owning run's recorded judge identity. A reader without an own
run, or a run with no score for that judge yet, uses the newest recorded identity
in the selected group. A caller's `run_id` filter limits candidates, independently
of that identity choice. Hidden learnings never appear in lists, searches, or
counts. Part-scoped learning reads use the resolving main goal; namespace-wide
reads still prefer that goal after higher-priority relevance criteria.

Board-level reads reuse the latest frontier, capturing one on the first read.
Call `frontier()` again to refresh. A previously created view stays fixed.
Plan resolution is retained per run and committed mark in the open board.
Repeated reads at unchanged marks do not replay journal history. Advancement
reuses committed replay checkpoints and folds only missing events; a cold or
historical prefix may require one reconstruction. Siblings reuse the fixed
resolved plan, so later ingestion cannot change their authority.
Lineage holds one shared frontier read scope per namespace and queries ancestor
IDs in breadth-first batches. Each requested ancestor is looked up at most once,
including missing parents and cycles; temporary sets are released on exit.

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
