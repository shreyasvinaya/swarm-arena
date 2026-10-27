# swarm_arena.llm

Needs the `llm` extra.

Install `swarm-arena[llm]` to import this optional package. The `cache` extra
adds response-cache backends; `langfuse` adds optional tracing dependencies.
Importing the package pins LiteLLM to its bundled model cost map for reproducible
prices. Importing the core package does not load these optional integrations.

```{eval-rst}
.. automodule:: swarm_arena.llm
   :members:
```
