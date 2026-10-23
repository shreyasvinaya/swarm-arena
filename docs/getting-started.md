# Getting started

`swarm-arena` runs teams of agents against a goal, prunes weak branches with
recorded reasons, and converges on ranked solutions. This example minimizes
Rosenbrock's function, `(1 - x)² + 100(y - x²)²`, entirely on your machine with
scripted managers. It needs no API key or model provider.

```text
                 ┌── Team A ──┬─ agent ─┬─ agent ──┐
                 │            ├─ agent ✗           ├──┐
  Goal ──────────┼── Team B ──┼─ agent ─┴─ agent ──┘  │
                 │            └─ agent ✗              ├──► Solution
                 └── Team C ✗ (pruned)                │
                                        (survivors) ──┘
```

## Install

Use Python 3.11 or newer on POSIX. For releases, install the package in a uv project:

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

To run the bundled example from a source checkout, use these two commands in the
repository root:

```sh
uv sync --all-extras --all-groups
uv run swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml
```

The CLI prints a worst-case estimate, the run's location, and finally
`done: rosenbrock 2 finalists`. `--yes` approves the estimate; omit it in an
interactive terminal to approve at the prompt. Even when the estimate is
unlimited, the configured round and action limits still apply.

Press Ctrl-C at the approval prompt or during `run` or `resume` to cancel.
The CLI finishes bounded cancellation cleanup, prints `interrupted` to stderr,
and exits with code 130. Cancelling approval leaves the run and artifact roots
unwritten. The terminal prompt uses nonblocking input and an event-loop signal
wakeup, so Ctrl-C also exits when the OS delivers SIGINT to a worker thread.
It needs no reader thread or buffered stdin lock and restores stdin's mode
and the interrupt handler afterward.

## Tools, check, and judge

The two `FunctionTool`s return data only. Gradient descent follows local
updates, while random search samples a square. Fixed seeds make this tutorial
repeatable; omitting `seed` gives fresh random samples. `BoundsCheck` rejects
missing, non-finite, or out-of-bounds results. The `ProgramJudge` minimizes `f`.
No extra dependencies are needed beyond the package.

```{literalinclude} ../examples/rosenbrock/rosen.py
:language: python
```

## Plan and limits

Each team contains two agents with different approaches. A `ScriptedManager`
selects requests explicitly for rounds 1 and 2. `SuccessiveHalving` prunes weaker
agents between rounds and records its reasons. Convergence selects two distinct
results and ranks them with a final judging batch.

The descent agents' second requests use `parents = "best"` to record lineage to
the best visible result. Their `x0` and `y0` remain explicit inputs: a parent does
not automatically change a function's arguments. `FunctionTool` stays data-only;
it does not apply file edits or parent file overlays.

The random agents each allow two replicates through `AgentSpec.replicates` and
supply one request per round. The allowance does not schedule requests by itself.
Their tools declare `deterministic = false`, permitting repeated requests.

```{literalinclude} ../examples/rosenbrock/plan.toml
:language: toml
```

The roots below are relative to the command's working directory. Keep all three
under `.swarm-arena/`; `max_rounds = 2` bounds this offline run.

```{literalinclude} ../examples/rosenbrock/run.toml
:language: toml
```

## Read the solution

Use the run directory printed by the CLI. Its
`goals/rosenbrock/solution.json` contains `status` and ordered `finalists`, each
with a `candidate` and its recorded `scores`. For this deterministic objective,
read the objective from the candidate's output data:

```python
import json
from pathlib import Path

run_dir = Path(".swarm-arena/runs/<run-id>")  # replace with the printed directory
solution = json.loads((run_dir / "goals/rosenbrock/solution.json").read_text())
print(solution["status"])
for finalist in solution["finalists"]:
    candidate = finalist["candidate"]
    print(candidate["id"], candidate["team"], candidate["output"]["data"]["f"])
```

Finalists are in final ranking order. Their `scores` include historical scores
as well as final scores; the first array entry is not necessarily the final
score. The final ranking batch is identified by committed main-goal `score`
events with `round = 0`, using their score IDs to select the corresponding
entries in `scores`. The example's end-to-end test verifies that relationship.

The same run is available through the public Python API, from the repository root:

```python
import anyio
from swarm_arena import Plan, RunConfig, Swarm


async def main() -> None:
    config = RunConfig.load("examples/rosenbrock/run.toml")
    results = await Swarm(config).run(Plan.load("examples/rosenbrock/plan.toml"))
    for candidate in results["rosenbrock"]:
        print(candidate.id, candidate.output.data["f"])


anyio.run(main)
```

`Plan.load` captures the plan directory so `rosen:*` imports resolve both in the
engine and function subprocesses. No `PYTHONPATH` setup is needed.

## Inspect, stop, resume, and reconcile

Replace `<run-id>` below with the run ID from the start line. For a live run,
`stop` requests a graceful stop; `--now` requests immediate cancellation. A
completed run has no live owner to stop. Resume a stopped or interrupted run
once its previous owner has released the lease or the lease has expired.

```sh
uv run swarm-arena status .swarm-arena/runs/<run-id>
uv run swarm-arena stop .swarm-arena/runs/<run-id>
uv run swarm-arena stop --now .swarm-arena/runs/<run-id>
uv run swarm-arena resume .swarm-arena/runs/<run-id>
uv run swarm-arena reconcile .swarm-arena/runs/<run-id>
```

Resume uses the run's saved `run.toml`. To supply a replacement configuration,
use `resume RUN_DIR --config RUN_TOML`; `--force` permits an explicit ownership
takeover. See [Limits and safety](limits-and-safety.md) before changing limits or
forcing takeover.

Reconcile uses the same saved-configuration addressing and accepts
`reconcile RUN_DIR --config RUN_TOML`. Run directories and run roots must be
writable only by the operator: a writer can change what resume executes and
make resume or reconcile send the master key named by `master_key_env` to the
gateway URL in that configuration. On shared storage, pass `--config` from a
trusted copy; gateway continuity checks refuse a changed URL. See
[post-run reconciliation and storage trust](limits-and-safety.md#after-the-run-reconcile).

## With an API key

Install the `llm` extra and configure a model alias and its provider credentials.
An agent's manager changes in one line:

```toml
manager = {path = "swarm_arena.llm:AgentsSDKManager", config = {model = "opus"}}
```

The supplied `examples/rosenbrock/plan_llm.toml` uses Opus and Sonnet managers
and adds a GPT/Haiku advisory panel beside the programmatic ranking judge.
`run_llm.toml` supplies the aliases, a one-million-token cap, two rounds and zero
tool reservations because these tools make no model calls. It has no USD cap.
Set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in your environment, then run:

```sh
uv sync --extra llm
uv run swarm-arena run --yes examples/rosenbrock/run_llm.toml examples/rosenbrock/plan_llm.toml
```

The [example walkthrough](https://github.com/shreyasvinaya/swarm-arena/blob/main/examples/rosenbrock/README.md#with-api-keys) includes
offline `FakeModel` tests and the opt-in live test. All ordinary tests are offline.
Final judging reserves serialized input at call time; output-only final slices
cannot guarantee that arbitrary candidate text will fit the remaining budget.

For your own programs, see [Try your own scripts offline](own-files.md#try-it-offline)
and [Writing a connector](writing-a-connector.md).
