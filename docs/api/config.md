# swarm_arena.config

```{eval-rst}
.. automodule:: swarm_arena.config
   :members:
```

## Load and save run settings

`RunConfig` is available from `swarm_arena`. It validates human-controlled
settings without starting an executor, contacting a provider, or reading secret
values. Relative storage roots retain their original spelling.

```python
from swarm_arena import RunConfig
from swarm_arena.config import ModelCatalog, ModelSpec, SecretsPolicy

config = RunConfig(
    secrets=SecretsPolicy(allow=["ANTHROPIC_API_KEY"]),
    models=ModelCatalog(
        {
            "opus": ModelSpec(model="anthropic/claude-opus-5-5", roles=["judge"]),
        }
    ),
)
text = config.dumps()
restored = RunConfig.loads(text)
assert restored == config
# Save text to run.toml, then load it with RunConfig.load("run.toml").
```

Model aliases have explicit role permissions, checked by
`config.models.resolve("opus", "judge")`. Cache reads and writes have separate
switches for each role. An omitted cache disables response caching; S3 and Redis
backends require `s3_path` and `redis_url_env`, respectively.

## Planner catalog rules

Each `CatalogEntry` supplies a component import path, fixed configuration, and a
mapping of settable field names to supported JSON-schema property definitions.
Schemas are checked when the entry is validated. `settable_model()` produces a
strict, frozen override model that rejects unknown keys. Omitted fields default
to `None`, including fields whose schema specifies a default. Explicitly supplied
values must satisfy their property schemas. The supported subset and reserved
parameter names are described in {ref}`ScriptTool parameters <scripttool-parameters>`:
`number` inputs must be finite; an array without `items` is untyped. Integer float
bounds must be integral int64 values, enum cannot be combined with numeric bounds
or pattern, and patterns use Rust regex syntax (no lookaround or backreferences; `\d` is Unicode-aware).

`PlannerCatalog` rejects duplicate names. For entries with settable fields it
imports the component class during validation and rejects fields listed in that
component's human-only settings. Entries without settable fields need no import.
`match()` compares the component kind and import path exactly, then removes
explicitly settable top-level fields from the reference configuration and compares
the remaining fixed settings. JSON types and values must agree recursively:
`True`, `1`, and `1.0` are distinct, including inside lists and nested mappings.
Mapping order does not matter; list order does. A nested field remains fixed even
if its name matches a settable top-level field. Matching performs no imports or
I/O and does not validate defaults or settable values; full component validation
belongs to the current `validate_plan` admission checks. See
[Planning modes](../planning-modes.md#frozen-files-and-catalog-defaults) for the
engine's default-aware catalog matching and history-based manifest trust.

Load errors report the configuration path and schema field names, aliases and
list/tuple indices; human settings show unknown model keys as typo identifiers,
component configs use `<field>`, and free-form mapping or `Any` segments use `<…>`. Rejected
values, schema defaults, import exception contents, and data-derived mapping keys
are omitted from diagnostics.

Validation diagnostics contain at most 20 error details in Pydantic order,
followed by `and N more errors` when truncated, including in component `ConfigError`
summaries. The omitted count includes errors hidden by explicit nested validation.
Any segment on which matching schema paths disagree is hidden, including string identifiers.

## Portable values and execution limits

Free-form goal inputs, goal metadata, component configuration, round overrides,
and tool output data accept finite JSON values with string keys and strict UTF-8
text. TOML-bound data rejects nulls at construction so persistence cannot silently
drop values. Tool outputs and other JSON-only payloads may contain JSON null.
Optional typed fields, such as an unset gateway, remain optional and are omitted
when saving. A component instance can omit typed fields whose default is
`None`; an explicit `None` that would replace a non-null default is refused.

JSON containers are limited to 100 levels of nesting. Edit text and compiled
string parameters also reject lone Unicode surrogates before encoding a request.
File manifests require normalized workspace paths and 64 lowercase hexadecimal
SHA-256 digests.

Budgets, usage counters, estimates, prices, and resource requests must be finite
and nonnegative. Timeouts and synchronization intervals must be strictly positive;
shutdown grace periods may be zero. A reported tool usage count or cost below
zero makes its result invalid.

`Plan.load()` and `RunConfig.load()` turn file, encoding, syntax, and validation
failures into `ConfigError`. TOML syntax diagnostics retain line and column
numbers while omitting source snippets. Component import errors name only the
import path. Sanitized validation errors support pickling and deep copying.

For a tool `Job`, a supplied `stage.request_json` must agree with its `request`.
Setting `stage.request_json=None` intentionally omits `swarm/request.json`; tools
that read this file must supply a request document.
