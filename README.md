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
`swarm_arena.testing` conformance kit. Model-driven managers and judge panels are
available with the `llm` extra.

## Quickstart

For releases, install the package in a uv project:

```sh
uv add swarm-arena
```

**Before the first release**, the PyPI package is a name-reservation placeholder
without this CLI or API. Install the development source instead:

```sh
uv add "swarm-arena @ git+https://github.com/shreyasvinaya/swarm-arena"
```

Contributors can clone the source and set up its environment:

```sh
git clone https://github.com/shreyasvinaya/swarm-arena && cd swarm-arena && uv sync
```

From a source checkout (Python 3.11+, POSIX), run the Rosenbrock swarm without an
API key or model provider:

```sh
uv sync --all-extras --all-groups
uv run swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml
```

`uv run swarm-arena plan examples/rosenbrock/run.toml examples/rosenbrock/plan.toml` validates and estimates a plan; [LLM and together modes](docs/planning-modes.md#from-the-command-line) generate journal-linked plan files.

Two teams compare gradient descent and random search over at most two rounds.
The CLI prints an estimate, the run directory, and `done: rosenbrock 2 finalists`.
Read `<run-dir>/goals/rosenbrock/solution.json` for the ordered results; run data
stays under `.swarm-arena/`. See [Getting started](docs/getting-started.md) for the
included source, plan, limits, Python API, and `status`/`stop`/`resume` commands,
or the [example README](examples/rosenbrock/README.md) for a short walkthrough.

With Anthropic and OpenAI API keys, the [LLM Rosenbrock example](examples/rosenbrock/README.md#with-api-keys)
uses model-driven managers and an advisory judge panel. Its offline exit tests use
isolated `FakeModel` conversations and check exact accounting, replay and final funding.

## Optional integrations

Install the optional model adapter dependencies, response-cache backends, or
Langfuse tracing dependencies as needed (combine extras for both model calls
and caching or tracing). Until M5's first release, install from the git source:

```sh
uv add "swarm-arena[llm] @ git+https://github.com/shreyasvinaya/swarm-arena"
uv add "swarm-arena[cache] @ git+https://github.com/shreyasvinaya/swarm-arena"
uv add "swarm-arena[langfuse] @ git+https://github.com/shreyasvinaya/swarm-arena"
```

## Run on a Ray cluster

Install the current development source and select the executor in `run.toml`:

```sh
uv add 'swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena'
# run.toml: executor = { path = "swarm_arena.ray:RayExecutor" }
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
# In another terminal: ssh -N -L 8265:127.0.0.1:8265 <head>
uv run swarm-arena submit --address http://127.0.0.1:8265 --working-dir . --yes run.toml plan.toml
```

Provision workers, module paths and authentication as described in
[Running on Ray](docs/running-on-ray.md), including the dashboard requirement for recovery.

An optional LiteLLM Proxy gateway gives each LLM-calling tool run its own virtual
key and records tool-internal calls, while plain tools stay unkeyed. The pinned
LiteLLM version supports **soft caps and estimated spend only**; keyed runs keep
their full USD reservation charged. Declare the tool's providers, accept soft
caps explicitly when using `max_usd`, and use `swarm-arena reconcile` for later
reports. See [The LLM gateway on the cluster](docs/running-on-ray.md#the-llm-gateway-on-the-cluster)
for setup, secret transport and worker network rules.

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
## The arena

Read committed results across runs, search evidence-backed learnings, and review
candidates from a laptop without a driver connection. From the source checkout
root, after the Quickstart environment setup, run this complete example:

```sh
ARENA_CONFIG=examples/rosenbrock/run.toml
ARENA_RUN_OUTPUT=$(uv run swarm-arena run --yes "$ARENA_CONFIG" examples/rosenbrock/plan.toml)
printf '%s\n' "$ARENA_RUN_OUTPUT"
# The CLI's "run ID at DIRECTORY" line supplies the generated run ID.
ARENA_RUN_ID=$(printf '%s\n' "$ARENA_RUN_OUTPUT" | awk '$1 == "run" {print $2}')
uv run swarm-arena leaderboard "$ARENA_RUN_ID" --config "$ARENA_CONFIG"
ARENA_LEARNING_ID=$(uv run swarm-arena learnings add --namespace toy --title "Rosenbrock baseline" --text "Compare gradient descent and random search." --author alice --config "$ARENA_CONFIG")
uv run swarm-arena learnings pin "$ARENA_LEARNING_ID" --namespace toy --author alice --config "$ARENA_CONFIG"
uv run swarm-arena board search "Rosenbrock baseline" --namespace toy --human --pinned --config "$ARENA_CONFIG"
uv run swarm-arena review export "$ARENA_RUN_ID" --out ".swarm-arena/review-$ARENA_RUN_ID" --config "$ARENA_CONFIG"
# For this demonstration, accept every exported candidate with score 1.
# For a real review, edit those verdict/score cells in a spreadsheet instead.
uv run python - "$ARENA_RUN_ID" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(".swarm-arena") / f"review-{sys.argv[1]}" / "review.csv"
with path.open(newline="") as stream:
    reader = csv.DictReader(stream)
    columns, rows = reader.fieldnames, list(reader)
for row in rows:
    row["verdict"], row["score"] = "accept", "1"
with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
PY
uv run swarm-arena review import "$ARENA_RUN_ID" ".swarm-arena/review-$ARENA_RUN_ID/review.csv" --reviewer alice --config "$ARENA_CONFIG"
uv run swarm-arena leaderboard "$ARENA_RUN_ID" --by human:alice --config "$ARENA_CONFIG"
```

All arena commands accept `--config PATH` (default `run.toml`). Run selectors
are run IDs; `status`, `stop` and `resume` use a local run directory. See
[The arena](docs/arena.md#command-line) for search filters, human curation,
provenance tracing, cache rebuilding and output formats.

## License

MIT. See [LICENSE](LICENSE).
