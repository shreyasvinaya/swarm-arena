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
            argv=["python", "-m", "my_connector.route"],
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
  `collect()` raises, the run gets status `error` and the traceback goes to the
  run's stderr log.

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

For your own `Tool`, `collect()` decides what is kept: the core sees only the
`ToolOutput` that it returns. The executor uploads only the files in
`ToolOutput.files` (it reads each path in the workspace and stores the file by
its SHA-256). So a file that the program writes but `collect()` does not list
is lost, and so are `usage` and `failed_provider` if `collect()` does not set
them. The `RouteTool.collect()` above keeps `data` only.

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

```python
# my_connector/scoring.py
from collections.abc import Sequence
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
        command = Command(argv=["python", "-m", "my_connector.validate"], python=self.config.python)
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

- `check_tool(tool, inputs, *, goal, parents, edits, executor, timeout_s)`
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
  failed run: `collect()` may raise there, but it must not report `ok`. A
  refused edit raises a `ValueError` that names the path and the reason; a
  refused base-parent overlay gives a `UserWarning`.
- `check_check(check, candidate, *, goal, executor)` runs one check and returns
  its `CheckResult`. `feedback` must be `"pass_fail"` or `"reasons"`. The
  result is validated again in strict mode, so a value that only lax
  validation would repair, such as `passed="no"`, is an `AssertionError`.
- `check_judge(judge, goal, candidates, *, executor)` judges one batch and
  returns the stamped scores. Each score is validated again in strict mode, as
  for a check result. The judge identity must be
  stable: `rubric_hash()` gives the same nonempty `str` on two calls and on a
  second build, and `panel_hash()` the same `str`. `anchor_count()` must return
  an `int` of 0 or more, `providers()` a frozenset of `str`, and `models()` a
  tuple of `str`. Each `ctx.judge_error` gives a `UserWarning`
  `"judge error for <id>: <reason>"`, and a candidate with neither a score nor
  a judge error gives a `UserWarning` too. A score for a candidate that is not
  in the batch, or two scores for one candidate, raise `AssertionError`.
- `make_candidate(output, *, request, tool, **fields)` makes a `Candidate` with
  local defaults. With `tool=`, it gets that tool's identity, so `check_tool`
  applies its overlay when it is the first parent.

Install the connector in the interpreter that runs the tool, for example with
`uv sync` or `pip install -e .`. The program runs with the workspace as its
working directory and a fresh environment without `PYTHONPATH`, so the
`pythonpath` setting of pytest or a source checkout is not enough: the driver
imports your package, but `python -m my_connector.route` fails.

One complete test module, run with `pytest` (the `anyio` plugin comes with
swarm-arena):

```python
# tests/test_conformance.py
import pytest
from swarm_arena import Goal
from swarm_arena.testing import check_check, check_judge, check_tool, make_candidate
from swarm_arena.tool import RunStatus, ToolOutput

from my_connector.scoring import FewestSteps, RouteIsValid
from my_connector.tools import RouteTool

pytestmark = pytest.mark.anyio
GOAL = Goal(id="ethanol", namespace="my-connector", inputs={"target": "CCO"})


async def test_tool_runs_and_follows_the_contract() -> None:
    out = await check_tool(RouteTool(), {"target": "CCO", "max_steps": 2}, goal=GOAL)
    assert out.status == RunStatus.OK
    assert out.data == {"target": "CCO", "steps": ["step 1", "step 2"]}


async def test_check_passes_a_route_and_fails_an_empty_one() -> None:
    route = make_candidate(ToolOutput(status=RunStatus.OK, data={"steps": ["step 1"]}))
    empty = make_candidate(ToolOutput(status=RunStatus.OK, data={"steps": []}))
    assert (await check_check(RouteIsValid(), route, goal=GOAL)).passed
    result = await check_check(RouteIsValid(), empty, goal=GOAL)
    assert (result.passed, result.reasons) == (False, ["the route has no steps"])


async def test_judge_prefers_the_shorter_route_and_reports_bad_output() -> None:
    short = make_candidate(ToolOutput(status=RunStatus.OK, data={"steps": ["a"]}))
    long = make_candidate(ToolOutput(status=RunStatus.OK, data={"steps": ["a", "b"]}))
    scores = await check_judge(FewestSteps(), GOAL, [short, long])
    assert max(scores, key=lambda score: score.key).candidate_id == short.id
    broken = make_candidate(ToolOutput(status=RunStatus.OK, data={}))
    with pytest.warns(UserWarning, match=f"judge error for {broken.id}"):
        assert await check_judge(FewestSteps(), GOAL, [broken]) == []
```

Run the tests from the directory that your file globs are relative to: the kit
snapshots files relative to the current working directory, as a plan's files
are relative to the plan file. A tool, check, or judge can be given as an
instance or as a `Ref`, for example
`Ref(path="my_connector.tools:RouteTool", config={"timeout_s": 60})`, which is
how a plan names it.

### When a run does not end with status `ok`

The kit shows the last 2 KB of a run's stderr log for a tool run, or a command
that a check or judge runs, that does not end with status `ok`. If the kit call
then returns, each run with status `error` gives a `UserWarning` with that text.
If the call raises an error instead (a failed contract assert, or an error from
your own code that the kit passes on, such as a `KeyError` in `fingerprint()`,
`check()`, or `judge()`), the kit adds the tail of each run whose status is not
`ok` (`error`, `timeout`, `cancelled`, and so on) to that error as a note and
raises the error unchanged. To keep the whole workspace, with its logs, pass
your own executor:

```python
# tests/test_logs.py
from pathlib import Path

import pytest
from swarm_arena.executor import LocalExecutor
from swarm_arena.testing import check_tool
from swarm_arena.tool import RunStatus

from my_connector.tools import RouteTool


def stderr_log(root: Path) -> str:
    # Each workspace is root/<job id>-<random>/, and its log is swarm/stderr.log.
    return next(root.glob("*/swarm/stderr.log")).read_text()


@pytest.mark.anyio
async def test_tool_with_kept_workspace(tmp_path: Path) -> None:
    executor = LocalExecutor(workspace_root=str(tmp_path), keep_workspaces=True)
    out = await check_tool(RouteTool(), {"target": "CCO"}, executor=executor)
    assert out.status == RunStatus.OK, stderr_log(tmp_path)
```
