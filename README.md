# swarm-arena

`swarm-arena` is an open-source Python library. It runs a **swarm** of agents on one or more hard goals, each on its own goal → teams → agents tree, and returns the best answers it can prove.

```
                 ┌── Team A ──┬─ agent ─┬─ agent ──┐
                 │            ├─ agent ✗           ├──┐
  Goal ──────────┼── Team B ──┼─ agent ─┴─ agent ──┘  │
                 │            └─ agent ✗              ├──► Solution
                 └── Team C ✗ (pruned)                │
                                        (survivors) ──┘
```

1. A **goal** fans out to several **teams**.
2. Each team has several **agents**, and each agent does one dedicated task.
3. Weak branches get **pruned**.
4. The good results **converge** into one solution.

Status: under construction. v0.1 is built milestone by milestone.

## License

MIT. See [LICENSE](LICENSE).
