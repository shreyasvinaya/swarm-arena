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
solution inspection, and model-driven managers.

## With API keys

Install the `llm` extra, set `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in your
environment, then run the supplied LLM plan and config:

```sh
uv sync --extra llm
uv run swarm-arena run --yes examples/rosenbrock/run_llm.toml examples/rosenbrock/plan_llm.toml
```

The descent managers use the `opus` alias (Claude Opus 5.5); random-search
managers use `sonnet` (Claude Sonnet 5.5). An advisory `LLMJudge` panel uses
`gpt` (GPT-6) and `haiku` (Claude Haiku 5.5). `ProgramJudge` remains the ranking
judge, so final selection still minimizes the measured objective. These Claude
IDs are priced in the pinned LiteLLM 1.107 map. Provider credentials and access
are required for a live run; the local price map does not establish availability.

The config permits two rounds and one million tokens. The tools make no LLM
calls, so `default_run_reservation` is zero. There is no dollar cap in the
shipped config; set `limits.max_usd` for your own run after checking the estimate.
Never put keys in either TOML file.

The offline tests replace the catalog with priced fake models. Each scoped
manager conversation gets its own script; judge responses match each actual
batch. They check every call's journal record, exact whole micro-dollar charges,
candidate call links, panel identity, completed resume and a source-pinned trace.
A separate capped run uses an LLM ranking judge and converge manager, and checks
final order and final-slice spend. Final slices hold output terms; serialized
inputs are funded at call time. Oversized final inputs are refused before a
provider call with `no_result` reason `reserve_exhausted`.

From the source checkout, run only these test files:

```sh
uv sync --all-extras --all-groups
uv run pytest tests/e2e/test_rosenbrock_llm.py tests/e2e/test_llm_final_slice.py -v
```

The live smoke test is opt-in and skips unless both keys are present:

```sh
uv run pytest tests/e2e/test_rosenbrock_llm.py -m llm_live -v
```
