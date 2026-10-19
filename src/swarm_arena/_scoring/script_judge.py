"""Score candidates concurrently using separate external-script workspaces."""

from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Literal, cast

import anyio
from pydantic import ConfigDict, Field, ValidationError

from swarm_arena._model.goal import FileSet, Goal
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import _ConfigInputModel, config_or_fields
from swarm_arena._model.results import Candidate, Score
from swarm_arena._model.tooling import RunStatus
from swarm_arena._scoring.judge import Judge, JudgeContext
from swarm_arena._scoring.script_check import _preflight_files, _result_error
from swarm_arena._tooling.script_tool import script_command


class _JudgeResultFile(_ConfigInputModel):
    """Validate numeric judge output without coercing strings or booleans."""

    model_config = ConfigDict(strict=True, extra="ignore", frozen=True, allow_inf_nan=False)

    value: float | None = None
    key: list[float] | None = None
    verdict: str | None = None
    notes: str = ""


class ScriptJudge(Judge):
    """
    Score each candidate in its own workspace with a human-configured script.

    Candidates run concurrently through the executor's resource pool. Returned
    scores retain candidate order; malformed results affect only that candidate.

    Parameters
    ----------
    config : ScriptJudge.Config or None, optional
        Validated settings, mutually exclusive with keyword fields.
    **fields : Any
        Settings used to construct ``Config`` when no config is supplied.

    Examples
    --------
    >>> from swarm_arena import ScriptJudge
    >>> judge = ScriptJudge(command=['python', 'judge/score.py'], direction='min')
    >>> judge.config.direction
    'min'
    """

    name: ClassVar[str] = "script-judge"
    _human_only: ClassVar[frozenset[str]] = frozenset({"command", "python", "files", "env"})

    class Config(_ConfigInputModel):
        """
        Configure a script judge and its execution limits.

        Parameters
        ----------
        command : list[str]
            Nonempty argument vector executed in the staged workspace.
        python : str or None, default=None
            Interpreter replacing a leading ``python`` argument.
        files : FileSet, default=FileSet()
            Private files; a list of relative globs is accepted as shorthand.
        env : list[str], default=[]
            Environment variable names allowed into the subprocess.
        direction : {'min', 'max'}, default='max'
            Preference for value-only results; explicit keys always maximize.
        timeout_s : float or None, default=None
            Positive execution timeout, capped by the context's limit.
        grace_s : float, default=10.0
            Nonnegative shutdown grace period in seconds.
        resources : Resources, default=Resources()
            Executor scheduling and resource request.

        Examples
        --------
        >>> from swarm_arena import ScriptJudge
        >>> ScriptJudge.Config(command=['x']).direction
        'max'
        """

        model_config = ConfigDict(extra="forbid", frozen=True)

        command: Annotated[list[str], Field(min_length=1)]
        python: str | None = None
        files: FileSet = FileSet()
        env: list[Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]] = []
        direction: Literal["min", "max"] = "max"
        timeout_s: Annotated[float, Field(gt=0)] | None = None
        grace_s: Annotated[float, Field(ge=0)] = 10.0
        resources: Resources = Resources()

    def __init__(self, config: Config | None = None, /, **fields: Any) -> None:
        """Validate configuration without performing I/O."""
        super().__init__(config_or_fields(config, fields))

    @property
    def config(self) -> Config:
        """
        Return the validated script-judge settings.

        Returns
        -------
        ScriptJudge.Config
            Frozen configuration supplied at construction.

        Examples
        --------
        >>> from swarm_arena import ScriptJudge
        >>> ScriptJudge(command=['x']).config.command
        ['x']
        """
        return cast(ScriptJudge.Config, super().config)

    def preflight(self) -> None:
        """
        Require the judge's private files to be snapshotted.

        Raises
        ------
        RuntimeError
            If source globs have no manifest yet.

        Examples
        --------
        >>> from swarm_arena import ScriptJudge
        >>> ScriptJudge(command=['x']).preflight()
        """
        _preflight_files(self.name, self.config.files)

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        """
        Execute one script per candidate and return scores in candidate order.

        Parameters
        ----------
        goal : Goal
            Goal being judged; execution uses the context's goal files.
        candidates : Sequence[Candidate]
            Candidates to score in independent workspaces.
        ctx : JudgeContext
            Execution context and per-candidate error channel.

        Returns
        -------
        list[Score]
            Valid raw scores, omitting candidates with script or result errors.

        Raises
        ------
        Exception
            The first executor exception, unwrapped from its task group after
            sibling runs have been cancelled. Cancellation propagates too.

        Examples
        --------
        >>> import asyncio
        >>> from swarm_arena import Goal, Resources, ScriptJudge
        >>> from swarm_arena.board import Candidate
        >>> from swarm_arena.executor import Executor, Job
        >>> from swarm_arena.scoring import JudgeContext
        >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
        >>> class MemoryExecutor(Executor):
        ...     async def submit(self, job: Job) -> ToolOutput:
        ...         return ToolOutput(status=RunStatus.OK, data={'value': 0.25})
        ...     async def capacity(self) -> Resources | None:
        ...         return None
        ...     async def aclose(self) -> None:
        ...         return None
        >>> goal = Goal(id='g', namespace='example')
        >>> candidate = Candidate(id='c1', run_id='r', goal_id='g', team='t',
        ...                       agent='a', round=1, request=ToolRequest(),
        ...                       output=ToolOutput(status=RunStatus.OK))
        >>> context = JudgeContext(goal=goal, executor=MemoryExecutor(),
        ...                          artifact_root='/artifacts')
        >>> judge = ScriptJudge(command=['example'], direction='min')
        >>> [score] = asyncio.run(judge.judge(goal, [candidate], context))
        >>> (score.candidate_id, score.key, score.value)
        ('c1', (-0.25,), 0.25)
        """
        scores: list[Score | None] = [None] * len(candidates)

        async def run(index: int, candidate: Candidate) -> None:
            scores[index] = await self._score(candidate, ctx)

        try:
            async with anyio.create_task_group() as group:
                for index, candidate in enumerate(candidates):
                    group.start_soon(run, index, candidate)
        except ExceptionGroup as errors:
            first: Exception = errors
            while isinstance(first, ExceptionGroup):
                first = first.exceptions[0]
            raise first from None
        return [score for score in scores if score is not None]

    async def _score(self, candidate: Candidate, ctx: JudgeContext) -> Score | None:
        """Keep script and result failures local, letting executor errors escape."""
        out = await ctx.execute(
            script_command(self.config),
            files=self.config.files.manifest,
            candidate=candidate,
            resources=self.config.resources,
        )
        if out.status != RunStatus.OK:
            ctx.judge_error(candidate.id, f"judge script failed: {out.status}")
            return None
        try:
            result = _JudgeResultFile.model_validate(out.data)
        except ValidationError as error:
            nonfinite_key = any(
                item["loc"][0:1] == ("key",) and item["type"] == "finite_number"
                for item in error.errors(include_input=False, include_context=False)
            )
            reason = (
                "invalid judge result: key: must be nonempty and finite"
                if nonfinite_key
                else _result_error("judge", _JudgeResultFile, error)
            )
            ctx.judge_error(candidate.id, reason)
            return None
        if result.key is None and result.value is None:
            ctx.judge_error(candidate.id, "result needs key or value")
            return None
        try:
            if result.key is not None:
                score = Score(
                    key=tuple(result.key),
                    value=result.value,
                    verdict=result.verdict,
                    notes=result.notes,
                )
            else:
                factory = Score.minimize if self.config.direction == "min" else Score.maximize
                score = factory(
                    cast(float, result.value), verdict=result.verdict, notes=result.notes
                )
            return score.model_copy(update={"candidate_id": candidate.id})
        except Exception:
            ctx.judge_error(candidate.id, "invalid judge result: key: must be nonempty and finite")
            return None
