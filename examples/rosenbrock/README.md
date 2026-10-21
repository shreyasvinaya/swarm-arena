# Offline Rosenbrock swarm

Minimize `(1 - x)² + 100(y - x²)²` with two teams: gradient descent and random
search. Four scripted agents run for at most two rounds; `BoundsCheck` gates
results, `ProgramJudge` minimizes `f`, `SuccessiveHalving` records pruning reasons,
and convergence returns two distinct finalists. No API key or model is required.

From the repository root (Python 3.11+, POSIX):

```sh
uv sync --all-extras --all-groups
uv run swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml
```

Expect an estimate, `run <run-id> at <run-dir>`, and
`done: rosenbrock 2 finalists`. Read
`<run-dir>/goals/rosenbrock/solution.json` for the ordered finalists and recorded
scores. `scores` retains history; its first entry need not be the final score.
All storage roots are under `.swarm-arena/` in your current directory.

`plan.toml` resolves `rosen:*` relative to its own directory, including in tool
subprocesses. You can copy this whole folder elsewhere and run the installed
`swarm-arena run --yes run.toml plan.toml` from that folder without `PYTHONPATH`.

The two descent agents use different starting points. Their round-2 requests
record the best visible candidate as a parent, while explicit inputs continue
to determine their starting coordinates. Function tools return data only and do
not apply file edits or parent overlays. Random agents allow two replicates and
explicitly schedule one request per round; fixed seeds keep the example
repeatable. Remove the seeds for fresh samples.

Use the printed run directory to inspect or control a live/interrupted run:

```sh
uv run swarm-arena status .swarm-arena/runs/<run-id>
uv run swarm-arena stop .swarm-arena/runs/<run-id>
uv run swarm-arena resume .swarm-arena/runs/<run-id>
```

`stop --now RUN_DIR` requests immediate cancellation. `resume` uses the saved run
configuration and requires the previous owner to be gone or its lease expired.
See [Getting started](../../docs/getting-started.md) for a public Python API run,
solution inspection, and the future model-manager switch.
