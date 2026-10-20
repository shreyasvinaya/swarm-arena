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

## Later planning modes

Built-in LLM planning, interactive **together** planning, and **autonomous**
planning are M3 features. The public `Planner` interface and optional
`Swarm(..., replanner=...)` support custom revision producers now; they do not
install a model integration. Human review and the shared arena arrive in M2.
