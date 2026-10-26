# swarm-arena

`swarm-arena` is an open-source Python library. It runs a **swarm** of agents on one or more hard
goals, each on its own goal → teams → agents tree, and returns the best answers it can prove.

```text
                 ┌── Team A ──┬─ agent ─┬─ agent ──┐
                 │            ├─ agent ✗           ├──┐
  Goal ──────────┼── Team B ──┼─ agent ─┴─ agent ──┘  │
                 │            └─ agent ✗              ├──► Solution
                 └── Team C ✗ (pruned)                │
                                        (survivors) ──┘
```

Install with `uv add swarm-arena`; see [Getting started](getting-started.md#install)
for the source-install fallback until 0.1.0 is on PyPI.

```{toctree}
:maxdepth: 2

getting-started
concepts
own-files
writing-a-connector
planning-modes
scoring-and-review
limits-and-safety
running-on-ray
arena
versioning
roadmap
api/index
```
