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
values must satisfy their property schemas.

`PlannerCatalog` rejects duplicate names. For entries with settable fields it
imports the component class during validation and rejects fields listed in that
class's `_human_only` attribute. Entries without settable fields need no import.
`match()` compares the component kind, import path, and raw fixed configuration;
full component validation belongs to the later planning validation layer.

Load errors report the configuration path and schema field names, aliases and
list/tuple indices; human settings show unknown model keys as typo identifiers,
component configs use `<field>`, and free-form mapping or `Any` segments use `<…>`. Rejected
values, schema defaults, import exception contents, and data-derived mapping keys
are omitted from diagnostics.
