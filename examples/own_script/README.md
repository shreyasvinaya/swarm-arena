# Bring your own files and script

Fit `y = slope * x + intercept` with two editable model files and three
standard-library Python scripts. No connector subclass, model provider, or
network access is required.

From the repository root, after `uv sync --all-extras --all-groups`, run:

```sh
uv run swarm-arena run --yes examples/own_script/run.toml examples/own_script/plan.toml
```

The completion line is `done: line-fit 1 finalists`. Run records and artifacts
are stored under `.swarm-arena/` relative to your working directory. Inspect
`.swarm-arena/runs/<run-id>/goals/line-fit/solution.json` for the finalist,
scores, and lineage. The final coefficients are slope `2.0`, intercept `1.0`,
with public and private mean squared error `0.0`.

You can also run the same plan through the Python API from the repository root:

```python
import anyio

from swarm_arena import Plan, RunConfig, Swarm


async def _main() -> None:
    results = await Swarm(RunConfig.load("examples/own_script/run.toml")).run(
        Plan.load("examples/own_script/plan.toml")
    )
    print(results["line-fit"][0].output.data)


anyio.run(_main)
```

In round 1, `tuner-1` edits `problem/model_a.txt` to set the slope to `2.0`.
In round 2, `tuner-2` selects `parents = "best"` and changes only
`problem/model_b.txt` to set the intercept to `1.0`. Both tools have identical
base configurations, so the first candidate's slope overlay is applied before
the second request's intercept edit. `KeepTopTeams(k=1)` keeps the single team
and its agents, including the agent with no first-round request.

The files have these roles:

- `problem/model_a.txt` and `problem/model_b.txt`: editable TOML assignments,
  parsed without executing code. Whitespace and TOML comments are accepted;
  each file must contain its one named finite numeric coefficient.
- `problem/points.txt`: protected public training data, one whitespace-separated
  `x y` pair per line with no header. The goal glob snapshots this file, but the
  editable glob excludes it.
- `problem/run.py`: protected tool script. Reads `swarm/request.json` under
  `params`, multiplies both coefficients by `scale`, and writes `data` containing
  effective `slope`, `intercept`, and public MSE `fit_error` to
  `SWARM_ARENA_RESULT`. The schema supplies `scale=1.0` when requests omit it.
- `checks/check.py`: checks the nested `data` in `candidate/output.json` for
  finite numbers and nonnegative public error. Writes `passed` and `reasons`.
- `judge/score.py` and `judge/hidden/points.csv`: private scoring files. The CSV
  has an `x,y` header and held-out points distinct from the public dataset.
  The judge independently computes MSE from the candidate coefficients and
  writes `value`. `direction = "min"` prefers smaller errors.
- `plan.toml` and `run.toml`: component configuration and two-round run limits.

Check and judge files are staged only for their respective evaluation commands.
They are absent from tool workspaces. Local subprocesses run as your OS user;
this staging boundary does not provide strong process isolation. Use trusted
scripts and read the [isolation and import-shadowing guidance](../../docs/own-files.md).

Engine candidate `logs_ref` values are run-relative paths under
`goals/line-fit/logs/`. The conformance kit's `read_logs` reads content-addressed
kit results in the current Python process; it does not read these engine paths.

Verify overlays, hidden-file staging, and the scripts offline:

```sh
uv run pytest tests/e2e/test_own_script.py -v
```
