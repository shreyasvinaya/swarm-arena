# Writing a connector

A connector brings a domain program into swarm-arena: for example DeepRetro for
chemistry or physika for physics. It is a Python package with a `Tool` and, if
you need them, `Check` and `Judge` classes. The conformance kit
(`swarm_arena.testing`) lets you test in your own CI that they follow the
contracts below.

If your program is a script and your data are files, you may not need Python
classes at all: `ScriptTool`, `ScriptCheck`, and `ScriptJudge` are configured
in TOML (see [Use your own files and script](own-files.md)). This page is for
the case where you write the classes yourself.

The examples on this page are one small connector, `my_connector`, that plans a
synthesis route for a target molecule. Its files are:

```text
my_connector/
  __init__.py
  tools.py       # RouteTool
  route.py       # the program that RouteTool runs
  scoring.py     # RouteIsValid (a check) and FewestSteps (a judge)
  validate.py    # the program that RouteIsValid runs
tests/
  test_conformance.py
```

## A tool

A tool says **what** to run. The executor decides **where** it runs.

<!-- docs-test: file my_connector/tools.py -->
```python
# my_connector/tools.py
import json
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict
from swarm_arena.tool import Command, CompletedRun, RunStatus, Tool, ToolOutput, ToolRequest


class RouteInputs(BaseModel):
    """What the manager may set for one run."""

    model_config = ConfigDict(hide_input_in_errors=True)  # an LLM writes these values

    target: str
    max_steps: int = 5


class RouteTool(Tool):
    name = "route"
    description = "Plan a synthesis route for a target molecule (SMILES)."
    input_model = RouteInputs

    class Config(BaseModel):
        """Plan-level settings, written by a human in the plan."""

        model_config = ConfigDict(extra="forbid", frozen=True)

        python: str | None = None  # the program's own interpreter; None = the current one
        timeout_s: float | None = 600.0
        grace_s: float = 10.0

    @property
    def config(self) -> Config:
        return cast(RouteTool.Config, super().config)

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        return Command(
            argv=["python", str(Path(__file__).with_name("route.py"))],
            python=self.config.python,
            timeout_s=self.config.timeout_s,
            grace_s=self.config.grace_s,
        )

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        result = workspace / "swarm" / "result.json"
        if not result.is_file():  # the program crashed or was stopped before it wrote a result
            status = RunStatus.ERROR if run.status == RunStatus.OK else run.status
            return ToolOutput(status=status)
        partial = run.status in (RunStatus.TIMEOUT, RunStatus.CANCELLED)
        return ToolOutput(
            status=run.status, partial=partial, data=json.loads(result.read_text())["data"]
        )
```

<!-- docs-test: file my_connector/route.py -->
```python
# my_connector/route.py: the program. The core never imports it.
import json
import os

with open("swarm/request.json") as f:
    params = json.load(f)["params"]
steps = [f"step {i + 1}" for i in range(params["max_steps"])]  # your real planner goes here
with open(os.environ["SWARM_ARENA_RESULT"], "w") as f:
    json.dump({"data": {"target": params["target"], "steps": steps}}, f)
```

The rules:

- **A nested `Config`.** The plan's settings for the tool are validated with
  `Config`. Set `model_config = ConfigDict(extra="forbid", frozen=True)`, so a
  misspelled key is an error and not a silent default. The core builds the
  tool as `RouteTool(RouteTool.Config.model_validate(settings))`, and the base
  class keeps the validated model.
- **The `config` property override.** Return your own `Config` type, as above,
  so that `self.config.python` type-checks.
- **Inputs come from a manager**, often an LLM. Set
  `hide_input_in_errors=True` on the input model, so that an invalid value
  never shows in an error message or a log. (The kit also removes input values
  from the errors that it raises.)
- **No I/O in `__init__`.** Construction must not read files, import the
  program, or use the network. Put those checks in `preflight()`.
- **`command()` and `collect()` run next to the program**, on the worker that
  runs the tool. They get the workspace path: build every path from
  `workspace`, never from the process's working directory. (The kit runs them
  in your test's working directory, so it cannot find this mistake.) Only the
  `ToolOutput` and its files travel back to the driver. The core never imports
  the program.
- **`"python"` as the first argument** becomes `Command.python`, or the current
  interpreter when it is None. A tool is never started with `uv run`.
- **`collect()` always runs**, also after a timeout or a cancel (after SIGTERM
  and `grace_s`). Return the data you have with `partial=True` then. If
  `collect()` raises an ordinary exception or `SystemExit`, stderr records only
  `swarm-arena: collect failed: <Type>`, without the exception message or
  traceback. The run gets status `error`, preserving `timeout` or `cancelled`
  when already stopped. See the [executor outcome contract](own-files.md).

### The workspace and `swarm/result.json`

The job's workspace is the program's working directory. The goal's files keep
their relative paths, `swarm/request.json` holds the request, and the program
writes its result to the path in `SWARM_ARENA_RESULT`
(`<workspace>/swarm/result.json`):

```json
{"data": {"target": "CCO", "steps": ["step 1", "step 2"]}}
```

`data` is required. The data returned by `collect()` must be finite JSON (no
NaN and no infinity): `LocalExecutor` reports a violation as status `error`
(`timeout` or `cancelled` for a stopped run) with empty data, and under
`check_tool` it is an `AssertionError` on every path. `usage`, `files`,
`failed_provider`, and `schema_version` are optional in the file format that
the built-in `ScriptTool` reader uses. The full contract (request shape,
parents, edits, exit codes, partial results, the environment) is in
[Use your own files and script](own-files.md).

`LocalExecutor.Config.max_output_bytes` bounds the combined declared output
bytes (default 256 MiB), including during built-in collection before hashing.
Increasing that setting also increases the collection budget. Each result may
declare at most 4096 files, counting empty files; the executor rejects larger
custom results before serializing their models or reading their output files.
The built-in reader accepts at most 1 MiB of result JSON, with nesting limited
to 100 levels. Use output artifacts for larger script results. Hashing and
upload use reads of at most 1 MiB each. `FunctionTool` returns data only and
never uploads function-written artifacts on success, timeout or cancellation;
use `ScriptTool` or a custom `Tool` for output files.

For your own `Tool`, `collect()` decides what is kept: the core sees only the
`ToolOutput` that it returns. The executor uploads only the files in
`ToolOutput.files` (it reads each path in the workspace and stores the file by
its SHA-256). So a file that the program writes but `collect()` does not list
is lost, and so are `usage` and `failed_provider` if `collect()` does not set
them. The `RouteTool.collect()` above keeps `data` only.

### Tools that call an LLM

A tool that calls an LLM must declare `providers()`, returning the provider
names (for example `frozenset({"openai"})`); for `FunctionTool`, use
`providers=["openai"]`. With an LLM gateway, this declaration lets the driver
create one virtual key per tool run and inject the gateway base URLs and key.
The subprocess must use `OPENAI_BASE_URL` or `ANTHROPIC_BASE_URL` instead of
hard-coding a provider URL. A `Command.env` allowlist discovered only inside
`command()` cannot trigger driver key creation and, with a gateway, never
receives the real OpenAI or Anthropic credential. Driver-visible FunctionTool
allowlists can trigger keying, but declare providers explicitly for clarity.

Supply a positive USD reservation through `estimate()` or
`limits.default_run_reservation.usd`; the full amount stays charged even after
spend is read back. Never include virtual keys in results or logs. The pinned
LiteLLM profile supports soft caps and estimated spend only. See
[The LLM gateway on the cluster](running-on-ray.md#the-llm-gateway-on-the-cluster)
for configuration, secret transport, proxy caching/tracing and network rules.

### Optional methods and their defaults

| Method | Default | Override it to |
|---|---|---|
| `preflight()` | does nothing | check the interpreter, imports, or files before the first run |
| `is_deterministic(request)` | `True` | return `False` for a stochastic program, so that replicates are allowed |
| `editable()` | `()` | name the globs of files that a manager may edit or add |
| `estimate(request)` | `None` | give the expected seconds, tokens, or cost (`Estimate`) |
| `providers()` | `frozenset()` | name the LLM providers that the program calls |
| `fingerprint(output)` | SHA-256 of the canonical JSON of `output.data` | give equal results the same identity, for example by ignoring a timestamp |
| `extra_actions()` | `()` | give the manager more typed actions (`Action`) |
| `get_input_model()` | `type(self).input_model` | build the input model per instance |
| `get_description()` | `type(self).description` | build the description per instance |

## A check

A check runs right after each tool run. If it fails, the candidate is
**gated out**: it is kept, but it is never judged or selected. `feedback`
selects what managers see: `"pass_fail"` shows only passed or failed, and
`"reasons"` (the default) also shows `CheckResult.reasons`.

`ctx.execute(command, candidate=...)` runs a command through the executor, for
example a validator that needs RDKit in the tool's interpreter. The command's
workspace has the candidate's request, output, and files under `candidate/`,
which is read-only, and the command writes a raw JSON object (no `data`
envelope) to `swarm/result.json`. A check must **fail closed**: a crash or a
missing result is a failed check, never a pass.

## A judge

A judge scores a batch of candidates. Each `Score` has a `key`, a tuple that
is compared in order, where **a larger key is better**: use
`Score.maximize(x)` or `Score.minimize(x)`. Set `candidate_id` on each score.
Every key value must be finite. When you cannot score a candidate, call
`ctx.judge_error(candidate.id, reason)` and give it no score: the other
candidates are still judged. A reason is shown in logs, so it names the problem
and never holds secret values.

The check and the judge of `my_connector`, and the validator program that the
check runs:

<!-- docs-test: file my_connector/scoring.py -->
```python
# my_connector/scoring.py
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict
from swarm_arena import Goal, Score
from swarm_arena.board import Candidate
from swarm_arena.scoring import Check, CheckContext, CheckResult, Judge, JudgeContext
from swarm_arena.tool import Command, RunStatus


class RouteIsValid(Check):
    name = "route-valid"

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        python: str | None = None  # the interpreter that has the validator's dependencies

    @property
    def config(self) -> Config:
        return cast(RouteIsValid.Config, super().config)

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        command = Command(
            argv=["python", str(Path(__file__).with_name("validate.py"))], python=self.config.python
        )
        out = await ctx.execute(command, candidate=candidate)
        if out.status != RunStatus.OK:  # fail closed
            return CheckResult(check=self.name, passed=False, reasons=["validator failed"])
        return CheckResult(
            check=self.name,
            passed=out.data.get("passed") is True,
            reasons=out.data.get("reasons", []),
        )


class FewestSteps(Judge):
    name = "fewest-steps"

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        scores = []
        for candidate in candidates:
            steps = candidate.output.data.get("steps")
            if not isinstance(steps, list):
                ctx.judge_error(candidate.id, "the output has no steps list")
                continue
            score = Score.minimize(len(steps))  # fewer steps is better
            scores.append(score.model_copy(update={"candidate_id": candidate.id}))
        return scores
```

<!-- docs-test: file my_connector/validate.py -->
```python
# my_connector/validate.py: runs in the check's workspace, next to candidate/
import json
import os

with open("candidate/output.json") as f:
    output = json.load(f)
reasons = [] if output["data"].get("steps") else ["the route has no steps"]
with open(os.environ["SWARM_ARENA_RESULT"], "w") as f:
    json.dump({"passed": not reasons, "reasons": reasons}, f)
```

The core stamps each score with an id, the judge's name, and its
`rubric_hash()`. A score with a non-finite key is rejected as a judge error for
that candidate.

## Test the connector in CI

`swarm_arena.testing` runs your components offline, the way the core runs them,
and raises `AssertionError` (with the name of the method) when a component
breaks its contract:

- `check_tool(tool, inputs, *, goal, parents, edits, round=0, round_overrides=None,
  replicate=0, executor, timeout_s)`
  snapshots the tool's and the goal's files from the current working directory,
  calls `preflight()`, validates `inputs` with `get_input_model()` (an invalid
  input raises a `pydantic.ValidationError` without the input values), applies
  the edit rules and the base-parent overlay rule, runs one job on `executor`
  (a `LocalExecutor` by default), and returns the `ToolOutput`. It asserts that
  `get_input_model()` returns a pydantic model class, `get_description()` a
  `str`, `editable()` a sequence of `str` globs, `providers()` a frozenset of
  `str`, and `extra_actions()` a sequence of `Action` with unique names. After
  the run it asserts that the output is a valid `ToolOutput` whose files are in
  the artifact store, that `fingerprint()` gives the same nonempty `str` twice
  and does not change with `logs_ref` or the run times, that
  `is_deterministic()` returns a `bool`, and that `estimate()` returns `None` or
  an `Estimate`. It also calls `collect()` once on an empty workspace after a
  failed run: `collect()` may raise there, but it must not report `ok`.
  The probe runs in a cancellable daemon thread and fails the contract if it
  takes more than 30 seconds. A blocked probe cannot hold up interpreter exit.
  The run uses the smaller of the kit's `timeout_s` (60 seconds by default)
  and the tool's configured `timeout_s`; an unset tool timeout uses the kit
  cap. The probe's deadline is separate from the run timeout. A refused edit raises a `ValueError` that names the path and the reason; a
  refused base-parent overlay gives a `UserWarning`.
  Round overrides replace matching top-level config settings; nested values
  are replaced as a whole. The merged config is validated by the tool's own
  config model before file snapshots, preflight, input validation, staging,
  and timeout selection. Overlay identity uses the snapshotted base tool
  configuration with defaults expanded, without round overrides. Human-only settings,
  including their validation aliases, cannot be overridden. For `ScriptTool`
  these are `command`, `python`, `files`, `env`, and `editable`.
  `command()` and `collect()` receive the same round, original overrides,
  and replicate as `swarm/request.json`; `SWARM_ARENA_ROUND` carries that round.
- `check_check(check, candidate, *, goal, executor)` runs one check and returns
  its `CheckResult`. `feedback` must be `"pass_fail"` or `"reasons"`. The
  result is validated again in strict mode, so a value that only lax
  validation would repair, such as `passed="no"`, is an `AssertionError`.
  The goal id must equal `candidate.goal_id`, or the kit raises `ValueError`
  before running the check. Omitting `goal` selects the kit's `check` goal,
  which is valid only for candidates with `goal_id="check"`.
- `check_judge(judge, goal, candidates, *, executor=None, config=None)` judges one batch and
  returns the stamped scores. Every candidate's `goal_id` must match the
  supplied goal id, or the kit raises `ValueError` before running the judge.
  Each score is validated again in strict mode, as
  for a check result. The judge identity must be
  stable: `rubric_hash()` gives the same nonempty `str` on two calls and on a
  second build, and `panel_hash()` the same `str`. `anchor_count()` must return
  an `int` of 0 or more, `providers()` a frozenset of `str`, and `models()` a
  tuple of `str`. Each `ctx.judge_error` gives a `UserWarning`
  `"judge error for <id>: <reason>"`, and a candidate with neither a score nor
  a judge error gives a `UserWarning` too. A score for a candidate that is not
  in the batch, or two scores for one candidate, raise `AssertionError`.
- `make_candidate(output, *, request, tool, parents=(), goal=None, **fields)`
  makes a `Candidate` with local defaults. Pass the same `tool` and `goal` as
  to `check_tool`. It computes the full effective overlay from the accepted
  first parent plus the new edits, and fills `parent_ids` and `request.parents`.
  Existing request parent references are used when `parents` is omitted;
  supplying both requires them to agree. Later parents contribute read-only
  references. Identity, editable-path, protection and layout checks are the
  same as for the tool run; refused base overlays warn and are excluded.
  With no `tool`, inherited overlays are refused. `goal` supplies the default
  `goal_id`; the request supplies the default round. The request's round
  overrides are applied to `tool` before checking overlay eligibility, just as
  in `check_tool`. Identity uses the snapshotted base configuration, so different
  overrides for the same base retain one identity; effective settings still
  enforce editable paths and protection. Explicit candidate fields still
  replace defaults.
- `read_output_file(output, path)` reads a declared output file by its exact
  workspace-relative path and returns verified bytes.
- `read_logs(output)` returns `(stdout_bytes, stderr_bytes)` for that run,
  verified against their stored hashes. These are the complete logs retained
  by the executor, subject to its configured log limit. Missing or malformed
  log references raise `ValueError`; missing or corrupt stored artifacts
  raise a storage error. Both readers work after workspace cleanup and until
  this Python process exits. They read the kit's process-local artifact store.

The example commands locate each program beside its connector module with
`Path(__file__).with_name(...)`. The program still runs in the executor's
workspace, where its request and result paths resolve. Package those scripts
with the connector, and install it on each worker for remote execution.
A command written instead as `python -m my_connector.route` requires an
installed connector: a source checkout or pytest's import path alone is
insufficient because the workspace has no inherited `PYTHONPATH`.

The following conformance module uses only the installed core dependencies.
Run each asynchronous check with the driver below:

<!-- docs-test: file tests/test_conformance.py -->
```python
# tests/test_conformance.py
import warnings
from swarm_arena import Goal
from swarm_arena.testing import check_check, check_judge, check_tool, make_candidate
from swarm_arena.tool import RunStatus, ToolOutput

from my_connector.scoring import FewestSteps, RouteIsValid
from my_connector.tools import RouteTool

GOAL = Goal(id="ethanol", namespace="my-connector", inputs={"target": "CCO"})


async def test_tool_runs_and_follows_the_contract() -> None:
    out = await check_tool(RouteTool(), {"target": "CCO", "max_steps": 2}, goal=GOAL)
    assert out.status == RunStatus.OK
    assert out.data == {"target": "CCO", "steps": ["step 1", "step 2"]}


async def test_check_passes_a_route_and_fails_an_empty_one() -> None:
    route = make_candidate(ToolOutput(status=RunStatus.OK, data={"steps": ["step 1"]}), goal=GOAL)
    empty = make_candidate(ToolOutput(status=RunStatus.OK, data={"steps": []}), goal=GOAL)
    assert (await check_check(RouteIsValid(), route, goal=GOAL)).passed
    result = await check_check(RouteIsValid(), empty, goal=GOAL)
    assert (result.passed, result.reasons) == (False, ["the route has no steps"])


async def test_judge_prefers_the_shorter_route_and_reports_bad_output() -> None:
    short = make_candidate(ToolOutput(status=RunStatus.OK, data={"steps": ["a"]}), goal=GOAL)
    long = make_candidate(ToolOutput(status=RunStatus.OK, data={"steps": ["a", "b"]}), goal=GOAL)
    scores = await check_judge(FewestSteps(), GOAL, [short, long])
    assert max(scores, key=lambda score: score.key).candidate_id == short.id
    broken = make_candidate(ToolOutput(status=RunStatus.OK, data={}), goal=GOAL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        assert await check_judge(FewestSteps(), GOAL, [broken]) == []
    assert any(
        isinstance(warning.message, UserWarning)
        and f"judge error for {broken.id}" in str(warning.message)
        for warning in caught
    )
```

```python
import runpy

import anyio

checks = runpy.run_path("tests/test_conformance.py")


async def run_checks() -> None:
    await checks["test_tool_runs_and_follows_the_contract"]()
    await checks["test_check_passes_a_route_and_fails_an_empty_one"]()
    await checks["test_judge_prefers_the_shorter_route_and_reports_bad_output"]()


anyio.run(run_checks)
```

Run the tests from the directory that your file globs are relative to: the kit
snapshots files relative to the current working directory, as a plan's files
are relative to the plan file. A tool, check, or judge can be given as an
instance or as a `Ref`, for example
`Ref(path="my_connector.tools:RouteTool", config={"timeout_s": 60})`, which is
how a plan names it.

### Connector CI

Keep the local conformance tests above and add a sequential Ray job. Install
the `llm` extra too when tests use `FakeModel`; scripted model turns need no
API key. In a source checkout, this GitHub Actions job runs the kit and the
Ray integration tests:

```yaml
jobs:
  connector-ray:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    env:
      RAY_ENABLE_UV_RUN_RUNTIME_ENV: "0"
      RAY_USAGE_STATS_ENABLED: "0"
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          python-version: "3.11"
      - run: uv sync --extra ray --extra llm --all-groups
      - run: uv run pytest -n 0 tests/testing/test_conformance.py tests/e2e/test_rosenbrock_ray.py -m "not s3" -v
```

The repository's shared `ray_cluster` fixture owns the head. Its **CI-only
two-CPU limit** keeps test load bounded. It launches the
equivalent of `uv run ray start --head --num-cpus=2 --include-dashboard=true
--disable-usage-stats --block` using the current Python interpreter, with
isolated ports and a short temporary directory. Readiness includes a successful
State API query. Teardown stops only its process group. Do not start a second
head in the job, use `ray stop`, or run cluster tests under xdist. The fixture
disables token authentication only for its loopback test head; that does not
test remote authentication. See [Running on Ray](running-on-ray.md) for token,
worker-image and module-path setup.

In a separate connector repository, provide an equivalent cluster fixture
(the test fixtures are not a public installed API), or connect to a cluster
your CI already owns. Put the connector in the uv project's dependencies;
until the first release, add the package with
`uv add "swarm-arena[ray,llm] @ git+https://github.com/shreyasvinaya/swarm-arena"`.
Use your connector test paths in the job. A Ray tool case takes this form,
using the `RouteTool` from this guide:

<!-- docs-test: skip needs a separately owned Ray cluster and its connector CI fixture -->
```python
import pytest

from my_connector.tools import RouteTool
from swarm_arena.ray import RayExecutor
from swarm_arena.testing import check_tool, read_logs

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.ray
async def test_tool_on_ray(ray_cluster: str) -> None:
    executor = RayExecutor()
    try:
        output = await check_tool(RouteTool(), {"target": "CCO"}, executor=executor)
        assert output.data["target"] == "CCO"
        stdout, stderr = read_logs(output)
        assert isinstance(stdout, bytes) and isinstance(stderr, bytes)
    finally:
        await executor.aclose()
```

Register `ray` under `[tool.pytest.ini_options]` in `pyproject.toml`, for
example `markers = ["ray: requires a Ray cluster"]`, including when using
`--strict-markers`. The AnyIO marker and backend fixture make the async test
runnable without `anyio_mode = "auto"`. This dedicated Ray job imports the
runtime directly: a missing extra or broken installation fails the job.
For mixed core/extras suites, skip only when the `ray` **distribution** is
absent; when installed, any failure to import `swarm_arena.ray` must fail.
The repository uses one shared `ray_dependency()` test helper for that rule;
`CI` does not change the decision.

Keep tests for `check_check` and `check_judge` alongside this tool case, and
use `check_manager` with the registered `FakeModel` catalog shown below for
model-driven managers. `FakeModel` can also supply scripted turns to an
`LLMJudge` (see the offline judge example below). These checks need no provider
credentials. A caller-supplied executor belongs to the caller and remains open
after a kit call; close it in `finally`.

Include assertions on file bytes, overlays and edits, both logs and stopped
results. `read_output_file` and `read_logs` read retained artifacts after the
worker workspace is gone. For cancellation, `Job.on_partial` receives only
data: file, log and report references are removed and no new artifacts are
uploaded. Shared-host data-only success does not demonstrate remote file
transport; use explicit filesystem-access denial or a real remote worker and
record which environment was exercised.

### Testing managers and judges offline

Install the optional SDK with
`uv add "swarm-arena[llm] @ git+https://github.com/shreyasvinaya/swarm-arena"`. Importing
`swarm_arena.testing` itself does not load the SDK; accessing `FakeModel` or
`FakeTurn` loads it. No API key is needed for scripted calls:

<!-- docs-test: requires llm -->
```python
import asyncio

from agents import Agent, Runner
from agents import RunConfig as SDKRunConfig
from swarm_arena.testing import FakeModel, FakeTurn

model = FakeModel([FakeTurn(text="ready")])
result = asyncio.run(
    Runner.run(
        Agent(name="offline", model=model),
        "Check the connector",
        run_config=SDKRunConfig(tracing_disabled=True),
    )
)
assert result.final_output == "ready"
assert model.calls[0]["input"]
```

For a component that resolves a catalog alias, register the instance and use
the corresponding `fake/` model in the run configuration:

<!-- docs-test: requires llm -->
```python
import asyncio

from swarm_arena import Goal, RunConfig, ScriptTool
from swarm_arena.config import ModelCatalog, ModelSpec
from swarm_arena.llm import AgentsSDKManager
from swarm_arena.testing import FakeModel, FakeTurn, check_manager

model = FakeModel(
    [
        FakeTurn(tool_calls=(("run_tool", {"inputs": {"x": 1}}),)),
        FakeTurn(text="done"),
    ]
)
model.register("connector")
config = RunConfig(
    models=ModelCatalog(
        {
            "manager": ModelSpec(
                model="fake/connector",
                roles=["manager"],
                prices=ModelSpec.Prices(input=1.0, output=2.0),
            ),
        }
    )
)
reports = asyncio.run(
    check_manager(
        AgentsSDKManager(model="manager", parallel_tool_calls=False),
        goal=Goal(id="g", namespace="offline"),
        tool=ScriptTool(command=["python", "-c", "print('ready')"]),
        rounds=1,
        config=config,
    )
)
assert reports[0].summary == "done"
assert model.calls[0]["model_settings"].parallel_tool_calls is True
```

The kit rebuilds a manager with `parallel_tool_calls=True` when its Config
supports that field. It binds the temporary run before invoking registered run
setups, and opens a fresh manager scope for each round. A normally returning
component that leaves LLM calls or reservations pending raises an `AssertionError`
naming the drain rule. On normal return, the kit also closes the round and
checks that its settled charge equals the scope's journaled LLM call costs,
using exact monetary arithmetic. A mismatch fails the same drain rule. If an
exception or owner interruption is already in flight, that outcome is preserved
with a value-free drain-rule note. Registered
terminal controls and genuine caller cancellation propagate. A manager
candidate linked to a call outside that round's scope fails the
`requested_by_call` rule. Cleanup settles outstanding reservations before
closing the owned services. Concurrent helper invocations use separate run
identities and services.

The offline kit has no engine deadline watcher for in-flight tool runs. If an
LLM manager's round timer fires during a core tool run, the tool finishes under
core rules: `round_timeout_s` does not cut the in-flight run in the kit;
`tool_timeout_s` bounds it.

`check_judge(judge, goal, candidates, config=config)` similarly opens a bound
judge scope with a funded tranche. Omitting `config` uses `RunConfig()` defaults.
Judge scripts can return their expected JSON as `FakeTurn.text` with a catalog
alias authorized for `judge`. Both helpers run registered setups once per run,
so real LiteLLM components receive the call logger too. A failing setup raises
`RuntimeError("run setup failed: <Type>")` from either helper, with value-free
diagnostics. Every metered fake call uses the real recorder, payload storage and reservation settlement;
the default turn consumes 100 input and 20 output tokens, costing $0.00014
at the example prices. Direct SDK calls without run metadata only replay turns.

Each scoped conversation (run, goal, team, agent, round, role and candidate)
has a separate replay queue. An explicit SDK `conversation_id` also separates
queues; unscoped calls without an ID consume one sequence. Concurrent agents
therefore do not alternate through one shared script. Tool-call IDs are unique
within the model, starting with `fake-run_tool-1` for `run_tool`.

Create and register a fresh model for each independent test or run to reset
history and replace consumed registrations. Registration is process-local;
test fixtures should restore previous registrations on teardown. Reusing an
instance under a new scoped run ID starts a new queue but retains `calls`.
Exhaustion raises `RuntimeError` containing `no turns left`; streaming raises
`NotImplementedError("FakeModel does not stream")`.

### Testing a round override

Pass round context explicitly to exercise a later round or replicate. Keep
the same base tool and request overrides when making a candidate from the
result:

```python
from swarm_arena import Goal
from swarm_arena.board import Candidate
from swarm_arena.testing import check_tool, make_candidate
from swarm_arena.tool import Tool, ToolRequest


async def run_round(tool: Tool, goal: Goal) -> Candidate:
    request = ToolRequest(inputs={}, round=3, round_overrides={"timeout_s": 10.0}, replicate=2)
    output = await check_tool(
        tool,
        request.inputs,
        goal=goal,
        round=request.round,
        round_overrides=request.round_overrides,
        replicate=request.replicate,
    )
    return make_candidate(output, tool=tool, goal=goal, request=request)
```

This example requires a tool whose config accepts `timeout_s` and whose input
model accepts empty inputs. A scalar setting such as `epochs` can be supplied
the same way when the connector's config declares it. The kit's timeout cap
still applies to an overridden timeout. Base overlays are accepted only when
their tool identity matches the snapshotted base config, with defaults expanded.
The kit runs one job; these arguments do not schedule additional rounds.

### When a run does not end with status `ok`

The kit shows the last 2 KB of a run's stderr log for a tool run, or a command
that a check or judge runs, that does not end with status `ok`. If the kit call
then returns, each run whose status is not `ok` gives a `UserWarning` with that text.
If the call raises an error instead (a failed contract assert, or an error from
your own code that the kit passes on, such as a `KeyError` in `fingerprint()`,
`check()`, or `judge()`), the kit adds the tail of each run whose status is not
`ok` (`error`, `timeout`, `cancelled`, and so on) to that error as a note and
raises the error unchanged. Read the stored logs without retaining a workspace:

<!-- docs-test: file tests/test_logs.py -->
```python
# tests/test_logs.py
from swarm_arena.testing import check_tool, read_logs
from swarm_arena.tool import RunStatus

from my_connector.tools import RouteTool


async def test_tool_logs() -> None:
    out = await check_tool(RouteTool(), {"target": "CCO"})
    stdout, stderr = read_logs(out)
    assert out.status == RunStatus.OK, stderr.decode(errors="replace")
```

```python
import runpy

import anyio

anyio.run(runpy.run_path("tests/test_logs.py")["test_tool_logs"])
```

For a declared file, use its path from `out.files`, for example
`read_output_file(out, "out/route.json")`. A file's display name may not be
unique; the path selects the file. No text decoding is implicit.

### Testing an edit chain

Use the same goal and tool for the run and candidate construction. For example,
with an existing goal and a tool whose editable globs include `problem/*.txt`:

```python
from swarm_arena import Goal
from swarm_arena.board import Candidate
from swarm_arena.testing import check_tool, make_candidate
from swarm_arena.tool import Tool, ToolRequest


async def edit_child(tool: Tool, goal: Goal, parent: Candidate) -> Candidate:
    request = ToolRequest(edits={"problem/b.txt": "3"})
    output = await check_tool(tool, {}, goal=goal, parents=[parent], edits=request.edits)
    return make_candidate(output, request=request, tool=tool, goal=goal, parents=[parent])
```

The child's overlay retains the accepted parent's edits, with the child's new
edits taking precedence at the same path. Passing that child as the next base
parent preserves the chain. Checks and judges supplied with the same goal see
the same effective files, without a manual overlay merge.


### Testing an LLM judge offline

The `llm` extra supplies `LLMJudge`; the conformance kit accepts its real model
catalog through `config=`. Register a fresh `FakeModel` for each test. This
example performs no provider request and verifies journaled call links:

<!-- docs-test: requires llm -->
```python
import asyncio
import json

from swarm_arena import Goal, RunConfig
from swarm_arena.config import ModelCatalog, ModelSpec
from swarm_arena.llm import LLMJudge
from swarm_arena.testing import FakeModel, FakeTurn, check_judge, make_candidate
from swarm_arena.tool import RunStatus, ToolOutput


async def check_offline_judge() -> None:
    goal = Goal(id="example", namespace="offline")
    candidates = [make_candidate(ToolOutput(status=RunStatus.OK), goal=goal)]
    row = {"index": 0, "value": 7.5, "verdict": "good", "notes": "Meets the rubric"}
    fake = FakeModel([FakeTurn(text=json.dumps({"scores": [row]}))])
    fake.register("m")
    config = RunConfig(
        models=ModelCatalog({"m": ModelSpec(model="fake/m", roles=["judge"], max_tokens=1000)}),
        settle_timeout_s=2,
    )
    scores = await check_judge(
        LLMJudge("Correctness from 0 to 10", ["m"]), goal, candidates, config=config
    )
    assert scores[0].value == 7.5
    assert scores[0].llm_call_ids


asyncio.run(check_offline_judge())
```

The kit checks that the judge leaves no reservation pending and that closed
charges equal journaled LLM costs. A contained terminal control fails the
conformance check; propagated controls and ordinary judge errors retain their
existing handling.

### Reporting calls made by your tool

A connector whose subprocess calls an LLM can write
`swarm/llm_calls.jsonl`
with one call per line, at most 64 KiB per line. See {ref}`Reporting LLM calls <reporting-llm-calls>`
for the schema, limits, candidate source labels and trace correlation variables.
The executor collects this optional file for tool jobs only; do not add its reserved
workspace path to `ToolOutput.files`. Keep aggregate tool usage in the result
file: individual report rows are informational and never create a second charge.
