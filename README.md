# swarm-arena

`swarm-arena` is an open-source Python library. It runs a **swarm** of agents on one or more hard goals, each on its own goal → teams → agents tree, and returns the best answers it can prove.

```text
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

## Install

Python 3.11+, Linux or macOS (POSIX). Install in a uv project:

```sh
uv init swarm-demo
cd swarm-demo
uv add swarm-arena
source .venv/bin/activate
```

Alternatively, use a plain virtual environment in your working folder:

```sh
python -m venv .venv
source .venv/bin/activate
pip install swarm-arena
```

If 0.1.0 is not on PyPI yet, use `uv add "swarm-arena @ git+https://github.com/shreyasvinaya/swarm-arena"` (add the desired extras inside the quoted requirement).
The earlier PyPI package is a name-reservation placeholder without this CLI or API.

Contributors can clone the source and set up its environment:

```sh
git clone https://github.com/shreyasvinaya/swarm-arena && cd swarm-arena && uv sync
```

| Extra | Adds |
| --- | --- |
| `swarm-arena[llm]` | Agents SDK managers, planners and judges. |
| `swarm-arena[cache]` | Response-cache backends. |
| `swarm-arena[ray]` | Ray cluster execution and Job submission. |
| `swarm-arena[s3]` | S3 run, arena and artifact storage. |
| `swarm-arena[langfuse]` | Opt-in Langfuse tracing export. |

Select the optional integrations you need:

```sh
uv add "swarm-arena[llm]"
uv add "swarm-arena[cache]"
uv add "swarm-arena[ray]"
uv add "swarm-arena[s3]"
uv add "swarm-arena[langfuse]"
```

## Quickstart (offline)

### Run hello-world

In the same activated shell, create a new empty folder:

```sh
mkdir hello-world
cd hello-world
```

Save these two files as `run.toml` and `plan.toml` in that folder.
They need only the installed core package: no checkout, API key or network.
One `ScriptedManager` drives a standard-library `ScriptTool` for one round.
The `ScriptJudge` awards a constant demonstration score of 1; it is not a
measure of greeting quality.

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

Run it with one command; `--yes` approves the printed estimate:

<!-- docs-test: run -->
```bash
swarm-arena run --yes run.toml plan.toml > hello-run-output.txt
cat hello-run-output.txt
```

### View results

One command displays the ranked results:

<!-- docs-test: run -->
```bash
swarm-arena leaderboard hello > hello-leaderboard.txt
cat hello-leaderboard.txt
```

For an optional local web viewer, run this command in your terminal. It runs
until Ctrl-C and is not executed by the docs tests; they run leaderboard above.

```sh
swarm-arena board serve hello
```

Default roots keep results under `.swarm-arena/`: read
`runs/<run-id>/goals/hello/solution.json` and `rounds/round-01.md` beside it.
The solution's `source_head` identifies its committed journal head.
Run this Python snippet in the same folder to read the greeting and save the
generated run ID for the review commands:

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

### Share a review

Export the run's review bundle, share it with a teammate, then import their
edited CSV. Export/import take the **run ID**, not its directory; `--out` names
a directory. The following commands reuse the ID saved above:

<!-- docs-test: run -->
```bash
swarm-arena review export "$(cat hello-run-id.txt)" --out review-bundle
```

The teammate edits `review-bundle/review.csv`: `verdict` is `accept` or `reject`,
`score` is numeric, and `notes` holds their explanation.

<details>
<summary>Try the review exchange offline</summary>

Run this Python snippet in the same folder to fill the single row with
`accept` and score `1`:

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

</details>

Import the edited CSV and read back the human score:

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
appropriate trusted configuration. See [The arena](https://swarm-arena.readthedocs.io/en/latest/arena.html) and
[Scoring and review](https://swarm-arena.readthedocs.io/en/latest/scoring-and-review.html) for details.

For the full Rosenbrock example and `AgentsSDKManager`, see
[Getting started](https://swarm-arena.readthedocs.io/en/latest/getting-started.html).

## Documentation

Read the [documentation](https://swarm-arena.readthedocs.io/) for planning,
connectors, scoring, limits, Ray execution and the shared arena.

## Status and versioning

The 0.x series is alpha. Names in the public modules' `__all__` are public;
removals require one minor release with a `DeprecationWarning`, and breaking
changes bump the minor version and appear in the changelog. Plug-in abstract
classes grow only optional methods with defaults and do no I/O in `__init__`;
stored formats carry a `schema_version`, load older versions through migrations
and refuse newer versions they cannot read.
See the [Versioning policy](https://swarm-arena.readthedocs.io/en/latest/versioning.html).

## Contributing

See [CONTRIBUTING.md](https://github.com/shreyasvinaya/swarm-arena/blob/main/CONTRIBUTING.md)
for development setup and checks.

## License

MIT. See [LICENSE](https://github.com/shreyasvinaya/swarm-arena/blob/main/LICENSE).
