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

## Status

Full offline swarm runs, scripted managers, pruning, convergence, and resumable
run journals work today, alongside tools, checks, judges, `LocalExecutor`, and the
`swarm_arena.testing` conformance kit. Model-driven managers arrive with the
`llm` extra (M3).

## Quickstart

From a source checkout (Python 3.11+, POSIX), run the Rosenbrock swarm without an
API key or model provider:

```sh
uv sync --all-extras --all-groups
uv run swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml
```

Two teams compare gradient descent and random search over at most two rounds.
The CLI prints an estimate, the run directory, and `done: rosenbrock 2 finalists`.
Read `<run-dir>/goals/rosenbrock/solution.json` for the ordered results; run data
stays under `.swarm-arena/`. See [Getting started](docs/getting-started.md) for the
included source, plan, limits, Python API, and `status`/`stop`/`resume` commands,
or the [example README](examples/rosenbrock/README.md) for a short walkthrough.

## Test one connector

Install from a clone (Python 3.11+, POSIX). In a fresh directory outside the clone,
replace `/path/to/agent-swarm` with the clone's absolute path and run:

```sh
uv venv --python 3.11 && uv pip install -e /path/to/agent-swarm
```

Run one function offline with `check_tool`. Save the function in an importable
module, `rosenbrock.py`, so `FunctionTool` can run it in a subprocess:

```python
def gradient_descent(lr: float, steps: int, x0: float = -1.5, y0: float = 2.0) -> dict[str, float]:
    """
    Minimize the Rosenbrock function with plain gradient descent.

    Parameters
    ----------
    lr : float
        Learning rate.
    steps : int
        Number of updates.
    x0, y0 : float, optional
        Starting coordinates, defaulting to -1.5 and 2.0.

    Returns
    -------
    dict[str, float]
        Final coordinates x, y and objective f.

    Examples
    --------
    >>> gradient_descent(0.001, 2000)["f"] < 12.5
    True
    """
    x, y = x0, y0
    for _ in range(steps):
        dx = 2 * (x - 1) - 400 * x * (y - x * x)
        dy = 200 * (y - x * x)
        x, y = x - lr * dx, y - lr * dy
    return {"x": x, "y": y, "f": (1 - x) ** 2 + 100 * (y - x * x) ** 2}
```

Save this driver beside it as `run_example.py` and run `uv run python run_example.py`
in the same directory:

```python
import anyio
from rosenbrock import gradient_descent

from swarm_arena import FunctionTool, Goal
from swarm_arena.testing import check_tool
from swarm_arena.tool import RunStatus


async def _main() -> None:
    out = await check_tool(
        FunctionTool(gradient_descent),
        {"lr": 0.001, "steps": 2000},
        goal=Goal(id="rosenbrock", namespace="e2e"),
    )
    assert out.status is RunStatus.OK
    assert out.data["f"] < 12.5
    print(out.data)


if __name__ == "__main__":
    anyio.run(_main)
```

The conformance kit uses `LocalExecutor` by default. See
[Try your own scripts offline](docs/own-files.md#try-it-offline) for a runnable
tool, check, and judge example, [Writing a connector](docs/writing-a-connector.md)
for custom components, or
[the end-to-end test](tests/e2e/test_single_run.py) for file edits, parent overlays,
and lexicographic scoring with private judge data.

Example: [Bring your own files and script](examples/own_script/README.md) demonstrates two-round edits, inherited overlays, and private scoring data.
## License

MIT. See [LICENSE](LICENSE).
