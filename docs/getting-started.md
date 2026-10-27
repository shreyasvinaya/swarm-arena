# Getting started

Run a small hello-world goal offline, view its result, then exchange a review
with a teammate. It needs only the core package, with no API key or model
provider. The later Rosenbrock example shows how `swarm-arena` runs teams of
agents, prunes weak branches with recorded reasons, and ranks solutions.

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
uv init swarm-demo
cd swarm-demo
uv add swarm-arena
source .venv/bin/activate
```

Alternatively, use a plain virtual environment in your working folder:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install swarm-arena
```

Use the [source-install fallback](https://github.com/shreyasvinaya/swarm-arena#install) until 0.1.0 is on PyPI;
the earlier package is a name-reservation placeholder without this CLI or API.

Contributors can clone the source and set up its environment:

```sh
git clone https://github.com/shreyasvinaya/swarm-arena && cd swarm-arena && uv sync
```

Optional extras are `llm` (Agents SDK managers, judges and planners), `cache`
(response caching), `ray` (cluster execution), `s3` (S3 storage), and `langfuse`
(tracing export). For example, add the LLM integration before the API-key
example later on this page:

```sh
uv add "swarm-arena[llm]"
```

The `llm` and `cache` extras currently require `litellm>=1.107.0rc2,<1.108`.
LiteLLM 1.107.0 final was not on PyPI when 0.1.0 was prepared; publication awaits
an owner decision to use the release candidate or wait for the final version.

## Run hello-world

In the same activated shell, create a new empty folder:

```sh
mkdir hello-world
cd hello-world
```

Save these two files as `run.toml` and `plan.toml` in that folder.
The one-round `ScriptedManager` calls a standard-library `ScriptTool`.
The `ScriptJudge` gives every result the constant demonstration score 1;
this shows the scoring path, not a measure of greeting quality.

<!-- docs-test: file run.toml -->
```toml
[limits]
max_rounds = 1
```

<!-- docs-test: file plan.toml -->
```toml
judges = [{name="hello", path="swarm_arena:ScriptJudge", config={command=["python", "-c", "import os; from pathlib import Path; Path(os.environ['SWARM_ARENA_RESULT']).write_text('{\"value\": 1}', encoding='utf-8')"], direction="max"}}]
[goal]
id = "hello"
namespace = "hello"
[[teams]]
name = "local"
rationale = "Say hello offline."
[[teams.agents]]
name = "greeter"
rationale = "Use the standard library."
tool = {path="swarm_arena:ScriptTool", config={command=["python", "-c", "import os; from pathlib import Path; Path(os.environ['SWARM_ARENA_RESULT']).write_text('{\"data\": {\"message\": \"hello world\"}}', encoding='utf-8')"]}}
manager = {path="swarm_arena:ScriptedManager", config={requests=[{round=1, inputs={}}]}}
```

Run it with one command. The redirection saves the CLI output for inspection;
`--yes` approves its printed estimate. Omit `--yes` to approve interactively.

<!-- docs-test: run -->
```bash
swarm-arena run --yes run.toml plan.toml > hello-run-output.txt
cat hello-run-output.txt
```

## View results

One command shows the hello namespace's ranked candidates:

<!-- docs-test: run -->
```bash
swarm-arena leaderboard hello > hello-leaderboard.txt
cat hello-leaderboard.txt
```

For the optional local web viewer, run the following command in your terminal.
It runs until Ctrl-C; this server command is not executed by the docs tests,
which execute the bounded leaderboard command above instead.

```sh
swarm-arena board serve hello
```

The defaults put the run, shared arena index inputs, and artifacts under
`.swarm-arena/` in this working folder:

```text
.swarm-arena/
  runs/<run-id>/
    run.toml
    manifest.json
    control/owner.json
    journal/e<epoch>-<last-seq>.jsonl
    goals/hello/
      plan.r0.toml
      rounds/round-01.md
      solution.json
  arena/
  artifacts/
```

Each run has one committed journal; the plan, round pages and `solution.json`
are derived views. A solution's `source_head` identifies the committed journal
head behind it. Read the successful greeting and save its actual generated
run ID for the review commands:

```python
import json
from pathlib import Path

run_line = next(
    line
    for line in Path("hello-run-output.txt").read_text(encoding="utf-8").splitlines()
    if line.startswith("run ") and " at " in line
)
run_dir = Path(run_line.split(" at ", 1)[1])
solution = json.loads((run_dir / "goals/hello/solution.json").read_text(encoding="utf-8"))
assert solution["status"] == "ok" and solution["source_head"]
assert len(solution["finalists"]) == 1
best = solution["finalists"][0]["candidate"]
assert best["output"]["data"] == {"message": "hello world"}
print(best["id"], best["output"]["data"])
Path("hello-run-id.txt").write_text(solution["run_id"] + "\n", encoding="utf-8")
```

## Share a review

Export the run's review bundle, share it with a teammate, then import their
edited CSV. Export/import take the **run ID**, not its directory; `--out` names
a directory. The following commands reuse the ID saved above:

<!-- docs-test: run -->
```bash
swarm-arena review export "$(cat hello-run-id.txt)" --out review-bundle
```

The teammate edits `review-bundle/review.csv`: `verdict` is `accept` or `reject`,
`score` is numeric, and `notes` holds their explanation. To try the exchange
offline, this block fills the single row with `accept` and score `1`:

```python
import csv
from pathlib import Path

path = Path("review-bundle/review.csv")
with path.open(encoding="utf-8", newline="") as stream:
    reader = csv.DictReader(stream)
    fieldnames = reader.fieldnames
    rows = list(reader)
assert fieldnames is not None and len(rows) == 1
rows[0].update(verdict="accept", score="1", notes="Greeting verified offline.")
with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
```

<!-- docs-test: run -->
```bash
swarm-arena review import "$(cat hello-run-id.txt)" review-bundle/review.csv --reviewer demo > hello-import.txt
cat hello-import.txt
swarm-arena leaderboard "$(cat hello-run-id.txt)" --by human:demo > hello-human-leaderboard.txt
cat hello-human-leaderboard.txt
```

Teammates need access to the existing run, arena and artifact roots and their
configuration. A review bundle does not upload or synchronize those stores.
For custom roots, pass `--config run.toml` to each arena/review command with the
appropriate trusted configuration. See [The arena](arena.md) and
[Scoring and review](scoring-and-review.md) for details.

## Rosenbrock

For a fuller source-checkout example, minimize Rosenbrock's function,
`(1 - x)² + 100(y - x²)²`, entirely offline. The three files shown below live
in `examples/rosenbrock/`; the docs harness copies that directory before
executing these commands. The hello-world recipe above needs no checkout.

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

The two `FunctionTool`s return data only: `gradient_descent(lr, steps, x0, y0)`
follows local updates, while `random_search(n, radius, seed)` samples a square.
Fixed seeds make this tutorial
repeatable; omitting `seed` gives fresh random samples. `BoundsCheck` rejects
missing, non-finite, or out-of-bounds results. The `ProgramJudge` minimizes `f`.
The scoring function returns `Score.minimize(f)`, so smaller objectives rank
higher. No extra dependencies are needed beyond the package.

```{literalinclude} ../examples/rosenbrock/rosen.py
:language: python
```

## Plan and limits

Two teams each contain two agents with different approaches. A `ScriptedManager`
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

<!-- docs-test: run -->
```bash
swarm-arena run --yes examples/rosenbrock/run.toml examples/rosenbrock/plan.toml > run-output.txt
cat run-output.txt
```

```python
import json
from pathlib import Path

run_line = next(
    line
    for line in Path("run-output.txt").read_text(encoding="utf-8").splitlines()
    if line.startswith("run ") and " at " in line
)
run_dir = Path(run_line.split(" at ", 1)[1])
solution_path = run_dir / "goals/rosenbrock/solution.json"
solution = json.loads(solution_path.read_text(encoding="utf-8"))
assert solution["status"] == "ok"
assert solution["finalists"]
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
    best = results["rosenbrock"][0]
    print(best.id, best.output.data)


anyio.run(main, backend="asyncio")
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
