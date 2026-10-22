# Planning modes

Manual plans run today through the CLI or `Swarm`. You choose the goal, tools,
managers, checks, judges, pruning policy, and convergence settings. The offline
`ScriptedManager` needs no model or API key. See the
[plan and Swarm API](api/swarm_arena.md), [planning API](api/planning.md), and
[configuration API](api/config.md).

## Manual TOML plans

From a source checkout, run the integrated Rosenbrock example:

```sh
uv run swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml
```

Omit `--yes` for interactive estimate approval. A directory of plans is loaded
in sorted filename order. The example's full plan is included directly here:

```{literalinclude} ../examples/rosenbrock/plan.toml
:language: toml
```

`Plan.load(path)` records the plan file's absolute parent directory. Imports and
file globs resolve relative to that directory. `Plan.loads(text, base_dir=...)`
can supply it explicitly; Python-created plans use the current directory.
`RunConfig` holds human-controlled limits and storage roots separately from the
plan. Relative storage roots follow the command's working directory.

```{literalinclude} ../examples/rosenbrock/run.toml
:language: toml
```

## From the command line

Validate a manual plan and print its worst-case estimate and admission warnings
without starting a run or asking for approval:

```sh
uv run swarm-arena plan examples/rosenbrock/run.toml examples/rosenbrock/plan.toml
```

Manual admission checks non-human catalog references before importing them and
snapshots files against the plan's directory. The estimate uses the same effective
round and convergence configurations as `run`; an uncapped unit prints `unlimited`.
Every admission error is printed on stderr with exit 1.

For model-generated planning, install the `llm` extra and supply a run configuration
with a planner-role model alias, a human-approved `planner_catalog`, and enough
token or USD budget for the planner. The catalog configuration in
[LLM planning](#llm-planning) is also usable here after saving it with
`config.dumps()` to `run.toml`. Create `goals/goal.toml` with just a `[goal]` table,
for example:

```toml
[goal]
id = "fit"
namespace = "optimization"
title = "Find a good solution using the approved catalog"
```

Then generate a plan, or review its TOML in the terminal before saving:

```sh
uv run swarm-arena plan --llm --model planner run.toml goals/goal.toml
uv run swarm-arena plan --together --model planner run.toml goals/goal.toml
```

`--instructions TEXT` adds trusted planning instructions. `-o OUT` selects an
output path; its resolved parent must be the goal file's directory, including
when the goal file is outside the working directory. Relative `-o` paths follow
the working directory. Defaults are `goals/plan.llm.toml` and
`goals/plan.together.toml`. An existing output is refused with exit 1 unless
`--force` is supplied; an output outside the goal directory is a usage error
(exit 2). Publication replaces the complete file atomically and checks for an
existing file again when publishing without `--force`. A directory or symlink
at the output path is refused before the planning session starts, even with
`--force`. New files use the process umask; replacement preserves the existing
file's permission mode. The complete file and its directory are synced before
publication. Cancellation during preparation of the output prevents publication;
the abandoned worker removes its temporary file when the pending I/O returns.

The goal and catalog files resolve against the goal file's directory, using the
same parent-directory convention as `Plan.load` even for a symlinked goal file. Source
files are frozen through the shared staging helper before the model call; its
[spool lifecycle](#llm-planning) also applies to CLI planning. Both generated and
edited drafts must pass catalog admission and preserve the frozen human goal.
Together mode uses the [terminal reviewer](#together), including feedback,
editing, safe retry diagnostics and EOF handling.

One planning session covers generation and all reviews. The CLI saves only after
validation and session cleanup succeed, then prints the output path and planning
session ID. The file retains `origin = "llm"` or `origin = "together"` and its
first-line session comment. Run it with, for example:

```sh
uv run swarm-arena run --yes run.toml goals/plan.llm.toml
```

The execution journal links to the planning session, so `trace` can show its
planner calls. The printed planning session ID is for accounting and provenance;
a planning session cannot be resumed. Initialization, settlement, EOF and invalid
draft failures do not save a plan or print success. Ctrl-C uses the shared CLI
interrupt owner: the first cancels, the second permits cleanup, and a third may
force exit. Normal interruption exits 130. Help and manual planning need no LLM
extra unless the human configuration itself references optional components.

## Manual Python plans

The same public models accept component instances as well as `Ref` objects.
Run this Python example from the source checkout's root with `uv run python`:

```python
import anyio
from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    FunctionTool,
    Goal,
    Limits,
    Plan,
    ProgramJudge,
    RunConfig,
    ScriptedManager,
    Swarm,
    TeamSpec,
)

plan = Plan(
    goal=Goal(id="rosenbrock", namespace="toy"),
    teams=[
        TeamSpec(
            name="descent",
            rationale="Follow the local gradient.",
            agents=[
                AgentSpec(
                    name="gd",
                    rationale="Search from a fixed starting point.",
                    tool=FunctionTool("examples.rosenbrock.rosen:gradient_descent"),
                    manager=ScriptedManager(
                        requests=[
                            {
                                "inputs": {"lr": 0.001, "steps": 1000, "x0": -1.5, "y0": 2.0},
                            }
                        ]
                    ),
                )
            ],
        )
    ],
    judges=[ProgramJudge("examples.rosenbrock.rosen:score")],
    converge=ConvergeSpec(top_n=1),
)


async def main() -> None:
    config = RunConfig(limits=Limits(max_rounds=1))
    results = await Swarm(config).run(plan)
    print(results["rosenbrock"][0].output.data["f"])


anyio.run(main)
```

`plan.dumps()` serializes a plan as TOML. `Plan.load` validates the file's
structure; execution admission also resolves components, snapshots their files,
and checks the whole plan. Public `validate_plan(plan, config, history=...)`
checks an **already frozen** plan and returns warnings or raises `ValueError`;
it refuses unresolved `FileSet` declarations. Use `Swarm.run` to admit source
plans rather than treating `validate_plan` as a snapshot operation.

## Provenance and revisions

`origin` distinguishes human plans from non-human proposals; `revision` and
`parent_revision` describe their history. Human-created plans default to
`origin="human"`. Loading a file **preserves an explicit non-human origin**:
saving and reloading a proposal does not give it human authority.

At initial admission the core keeps that origin and assigns revision zero.
For a `Planner` proposal it assigns the planner's origin, the next revision,
and the preceding revision. Planner subclasses cannot claim human origin.
The journal records admitted provenance, so proposal-supplied revision numbers
or origin fields cannot replace the core's decision.

`RunConfig.planner_catalog` is a human-maintained `PlannerCatalog` of permitted
component paths, fixed settings, and explicitly `settable` fields with schemas.
An unchanged reference at the same retained site may carry human authority
through admitted history. New or changed non-human references must pass catalog
authorization before the core imports or constructs their components. Fixed
comparison preserves JSON types: `True`, `1`, and `1.0` differ.

A non-human revision may change team structure, managers, tools, policy, rounds,
and permitted parameters within the catalog and human limits. It may add checks,
but must retain every earlier main check and judge unchanged and keep the ranking
judge. Existing part scoring is frozen; a new part inherits main scoring. The main
`GoalKey` stays fixed, and a part's files must be exact path-and-digest entries
from the main goal's frozen manifest. Removed team or agent addresses cannot be
reused. Names must be unique lowercase slugs in their scope, and each team and
agent needs a rationale.

Human-only component fields, including their validation aliases and nested
references, cannot become planner-settable. Round tool overrides cannot change
human-only fields, even in human plans. In non-human plans, overrides must also
be explicitly catalog-settable; restating a fixed field is refused.

## Frozen files and catalog defaults

Admission snapshots all component roots, including check and judge files,
nonempty `FileSet` defaults, and typed nested references in lists, tuples, and
mappings. Materialized defaults use canonical field names; explicitly supplied
file fields retain the input name or validation alias. Unrelated raw
configuration stays as written; a file-bearing container whose validation
would change its shape is refused instead of truncated.

A catalog entry may omit a component's declared `FileSet` default. A materialized
field is accepted only when its declaration equals that typed default, after
the import-free catalog match. This permits default files without making other
extra fields or different file declarations planner-settable.

Non-human component-file manifests (tools, checks, judges, and nested references)
are trusted only through admitted history. A supplied component manifest, even
for a catalog-default declaration, cannot authorize arbitrary
stored blobs. Without trusted history those files are snapshotted again from
the human source directory; missing sources fail admission. A human can
explicitly vouch for a saved snapshot by changing its origin to `"human"`.
Loading it alone never makes that change.

An initial plan's goal snapshot may be supplied. Later revisions must keep the
main goal's `GoalKey`. Non-human part manifests must be exact path-and-digest
subsets of the main manifest. These goal-file rules are separate from
component-file manifest trust.

Check- and judge-private files cannot appear in main or part goal files or in
tool-owned files, including alternate Unicode/case spellings. See
[Limits and safety](limits-and-safety.md) for the trust boundary and
[Use your own files and script](own-files.md) for the integrated file example.

Every GoalKey admitted by a goal remains reserved to that goal for the entire
run, including historical part keys and finished goals. Resume restores those
reservations and historical board reads from every admitted revision. A later
revision cannot assign another goal a key that appeared only in an older revision.

## LLM planning

Install the `llm` extra and configure a model alias with the `planner` role,
`limits.max_tokens`, and a human-maintained `planner_catalog`. If `max_usd` is
set, every used model alias needs a known price, from explicit catalog prices
or the bundled price map. Unused catalog aliases need no price. `LLMPlanner`
checks aliases before opening a standalone planning session or making a paid call.

```python
from swarm_arena.llm import LLMPlanner

# config contains your model aliases, limits, and permitted component entries.
plan = await LLMPlanner("planner").plan(goal, config, None)
```

The model selects catalog entry names and fills only their `settable` fields.
Entry kind, fixed settings, nested output shapes and authorization are checked
before execution. The core assigns `origin="llm"` and revision zero; model output
cannot assign human provenance. The human main goal, inputs and resolved files
are frozen before generation. A completed draft cannot change that identity.
Draft snapshots spool immutable bytes to a temporary file and upload each unique
digest once, one file at a time after validation. Resident staging memory scales
with the largest file, while the temporary spool scales with the unique source
bytes. The preparing worker closes the spool when preparation finishes, fails,
or observes abandonment. Cancellation cannot interrupt a synchronous step in
progress: a source read, a component import or validator, or a registered model
check. If such a step is blocked, the worker closes the spool on failure or
abandonment only once that step returns.

A direct initial Python call resolves relative goal and catalog files against
its current working directory. With history, the original admitted plan's
`base_dir` is used. Internal file-based callers supply that source directory
explicitly to the shared draft/admission helper; storage roots and generated
metadata do not select a source directory.

Outside a running scope, planning opens its own leased, journaled session with
registered run setups, a live timeout/stop clock, and a budget ledger. The active
allowance includes lease acquisition, even while the backend holds its response;
a stop or expiry during acquisition prevents setup and scope entry. Calls are
drained before return; stopped setup cannot yield a usable scope. Success and
interruption both retain committed usage for replay. No executor or board is
started for this session.

`plan.planning_session` links to that session and `plan.planner_call_ids` lists
its calls. Both properties are read-only. `plan.dumps()` writes a first-line
`# swarm-arena planning session: <run id>` comment. Loading TOML preserves the
validated session link and source directory, while call IDs become empty;
read the linked journal for those calls. Initial `plan_revision` records carry
this link even for queued goals, and replay restores the recorded provenance.

## Autonomous planning

Pass `replanner=LLMPlanner("planner")` to `Swarm(config, ...)`. The planner uses
the fixed history supplied by the engine: scores, untrusted learning titles,
pruning decisions, remaining budget, completed rounds and join failures.
Join-failure replans use the same admission path as ordinary revisions.

Replans keep all current plan-level checks and judges, ranking and policy;
additional catalog checks may be appended. `keep_teams` and `keep_parts` retain
named human components unchanged, including non-catalog ScriptTool teams and
part-specific checks, judges and ranking. A kept name must exist and cannot
also be newly defined. New parts inherit main scoring and can use only the
main goal's frozen files, never judge-private files.

An explicit new round schedule includes completed rounds, including completed
implicit rounds. Empty output rounds retain the current schedule. Every draft
passes shared authorization, file admission and component preflight before the
engine publishes an autonomous revision under the run's admission lock.
Rejected proposals keep the current revision. Fencing and caller cancellation
propagate as controls. In-run usage overflow stops the run through the budget
path; standalone planning raises the terminal accounting error.

An autonomous revision has no external `planning_session` link: its planner
call IDs belong to the current run journal. The r0 link remains unchanged.
Standalone cleanup records `goal_finished` before raising a terminal overflow.
Its reason distinguishes `planning complete`, `planner failed` (including an
invalid draft), `run budget`, `run_timeout_s`, `stop`, and `stop --now`; caller
cancellation records `stopped`. A fenced owner cannot append a finish event.
Trailing spaces and tabs after a saved planning-session comment are ignored.

## Together

`CollaborativePlanner` combines an `LLMPlanner` with a `Reviewer`. Each review
returns `Approve()`, `Feedback(text)`, or `Edited(plan)`. Feedback accumulates,
and the next model request includes the previous draft. The default limit is
five reviews; `max_iterations` must be a positive integer. Exhaustion raises
`RuntimeError("plan not approved after N reviews")` with the configured count.
The singular case says `plan not approved after 1 review`.
Direct Python together planning resolves relative goal files against cwd;
file-based callers retain the loaded goal file's directory through drafting
and review.

Using the same `goal` and catalog-backed `config` as LLM planning:

```python
from swarm_arena import Plan
from swarm_arena.llm import Approve, CollaborativePlanner, LLMPlanner


class ApprovingReviewer:
    async def review(self, plan: Plan) -> Approve:
        # Replace with your application's human review interface.
        return Approve()


planner = CollaborativePlanner(LLMPlanner("planner"), ApprovingReviewer())
plan = await planner.plan(goal, config, None)
```

The private terminal adapter is provided for the plan CLI. It displays TOML
and asks `Approve [a], feedback [f], edit [e]: `.
Feedback is one line. Editing invokes the executable selected by `editor`,
`VISUAL`, `EDITOR`, then `vi`, with one temporary-file argument and no shell.
Specify an executable path rather than a command with flags. An editor wrapper
must **exec** the editor (for example, `exec vi -n "$@"`) so cancellation reaches
the editor itself. Invalid TOML and editor failures print safe diagnostics and
retry, keeping the last edited file, including undecodable bytes, for repair.
EOF ends review with
`RuntimeError("terminal input ended")`. Edits load with the draft's `base_dir`.
Cancellation restores stdin's descriptor state, first asks the editor to terminate,
and reaps it, using a kill fallback after a one-second grace period. The terminal's
saved mode, including echo and canonical input, is restored after every editor exit.
Another Ctrl-C during the grace ends it immediately, kills and reaps the editor,
and lets the planning session finish its journal cleanup. The CLI also owns
repeated Ctrl-C at the prompt and during shutdown: two interrupts allow journal
cleanup to finish before exit 130; a third may force an immediate exit.

One metered planning session spans every draft and human review. Its run-time
allowance and stop/lease supervision remain active while awaiting input or the
editor; review does not restart the allowance. Each draft drains its accounting.
The result retains session provenance and is stamped `origin="together"`.
Edited plans pass the same catalog admission as generated drafts, preserve
their source directory, and cannot change the frozen human main goal. Its title,
description, metadata, namespace, id, inputs and file declarations are all fixed;
edited snapshots must still identify the original frozen files. A refused edit
prints a safe admission diagnostic to stdout and returns the same draft to the
reviewer for another review (including custom reviewers) without another paid
model call. The terminal adapter retains the edited text for repair; custom
reviewers must retain their own edits. Each refused edit counts toward `max_iterations`; giving feedback requests a new draft instead.
The terminal adapter clears retained edits when the collaborative session ends.
Approval cannot authorize a mutation of the offered draft: return
`Edited(plan)` to propose changes explicitly. Custom async reviewers must
cooperate with cancellation. Custom reviewer cleanup failures have type-only
diagnostics. Cleanup cannot replace an already escaping error or cancellation;
a cleanup failure after approval surfaces as an error.

Standalone planning journals remain readable for accounting and provenance.
They cannot be resumed: both `Swarm.resume` and `swarm-arena resume` refuse
with `run is a planning session; it cannot be resumed` before taking ownership
or appending any event, including for an interrupted planning session.
