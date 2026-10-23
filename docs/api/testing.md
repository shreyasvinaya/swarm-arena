# swarm_arena.testing

`FakeModel` and the frozen, keyword-only `FakeTurn` require the `llm` extra.
Both are lazy exports: importing this facade or listing its names does not
import `agents`. See [Testing managers and judges offline](../writing-a-connector.md#testing-managers-and-judges-offline)
for runnable scripts, catalog registration and conversation isolation.

`FakeModel.calls` records the six SDK inputs `system`, `input`,
`model_settings`, `tools`, `output_schema` and `tracing`. A metered turn uses
the pending reservation's catalog prices and the same core recorder as real
LLM calls; drain and owner supervision also apply to its payload uploads.

`OfflineHarness.run_rounds` honors `Limits.run_timeout_s`, including preparation
after lease acquisition. Reaching that deadline raises
`RuntimeError("offline harness stopped")`; owned services still receive bounded
cleanup. Set the limit to allow enough time for the rounds your test exercises.

```{eval-rst}
.. automodule:: swarm_arena.testing
   :members:
```
