# swarm_arena.scoring

```{eval-rst}
.. automodule:: swarm_arena.scoring
   :members:
```

Scores are available as `swarm_arena.Score`. Larger lexicographic keys rank
higher. Keys must be nonempty and contain only finite numbers.

```python
from swarm_arena import Score
from swarm_arena.scoring import CheckResult

loss = Score.minimize(0.25, notes="Lower loss is better")
assert loss.key == (-0.25,)
assert loss.value == 0.25
quality = Score.maximize(3.0, verdict="good")
passed = CheckResult(check="valid", passed=True)
```

`Score.identity` returns `(judge, rubric_hash, panel_hash)` for comparing
scores from the same judge configuration.


## Define checks and judges

Subclass `Check` to gate one candidate and `Judge` to score a batch. Both
validate their nested `Config` during construction without I/O; use `preflight()`
for readiness checks. A judge returns raw scores with `candidate_id` set.

```python
from collections.abc import Sequence
from typing import ClassVar

from swarm_arena import Goal, Score
from swarm_arena.board import Candidate
from swarm_arena.scoring import Check, CheckContext, CheckResult, Judge, JudgeContext


class Valid(Check):
    name: ClassVar[str] = "valid"

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        return CheckResult(check=self.name, passed="value" in candidate.output.data)


class Quality(Judge):
    name: ClassVar[str] = "quality"

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        return [
            Score.maximize(1.0).model_copy(update={"candidate_id": candidate.id})
            for candidate in candidates
        ]
```

`CheckContext.execute(command, files=manifest, candidate=candidate)` submits a
command job. Own files go at the workspace root, and the candidate's request,
output, effective files, and output files go under `candidate/`. The command
receives run/goal/candidate/round environment variables; team and agent are empty.
`JudgeContext` inherits this execution behavior and adds batch settings and
`judge_error(candidate_id, reason)`. The first reason for each candidate wins;
judges must supply diagnostics without secret values.

Both contexts accept `on_started`, `secrets_file`, a shared run `ids` source, and
`tool_timeout_s` (default `3600.0`). The command timeout defaults to that limit
and is capped by it. The context forwards the callback to `Job.on_started` and
lets executor exceptions propagate.

The core stamps copies of raw scores with ids and judge/rubric identity,
preserving candidate ids and existing panel/batch ids. The default rubric is a
SHA-256 digest of the canonical judge import path and configuration; default
panel identity, anchor count, provider set, and model tuple are empty/zero.
Scores created with `model_copy` or `model_construct` are checked again before ranking.

Script checks and judges accept an optional integer `schema_version` in their
result JSON. Omission means version 1; version 1 is supported, while a newer
version is refused with a clear version diagnostic. Both reject unknown fields,
including misspellings. Check output requires `passed`; judge output requires
`key` or `value`.

`swarm_arena.ScriptJudge.rubric_hash()` includes the command, interpreter, files,
environment names and scoring direction. Changing `timeout_s`, `grace_s` or
`resources` preserves the rubric identity. Custom judges should exclude settings
that only affect execution when defining their scoring identity.

`swarm_arena.ProgramJudge` records a scoring function's `SystemExit` as that
candidate's judge error and continues with the next candidate. KeyboardInterrupt
and cancellation propagate.
