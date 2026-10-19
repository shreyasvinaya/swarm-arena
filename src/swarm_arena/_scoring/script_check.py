"""Validate candidates with a human-configured external script."""

from typing import Annotated, Any, ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from swarm_arena._model.base import _sanitize_validation_error
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import _ConfigInputModel, config_or_fields
from swarm_arena._model.results import Candidate, CheckResult
from swarm_arena._model.tooling import RunStatus
from swarm_arena._scoring.check import Check, CheckContext
from swarm_arena._tooling.script_tool import script_command


class _CheckResultFile(_ConfigInputModel):
    """Validate check output strictly without exposing rejected values."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)

    passed: bool
    reasons: list[str] = []
    details: dict[str, Any] = {}


def _preflight_files(name: str, files: FileSet) -> None:
    """Reject unresolved globs before a script is executed."""
    if not files.snapshotted:
        raise RuntimeError(f"{name}: files are not snapshotted")


def _result_error(kind: str, model: type[BaseModel], error: Exception) -> str:
    """Render the first validation code with only safe schema locations."""
    first = _sanitize_validation_error(model, error).errors()[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    return f"invalid {kind} result: {location}: {first['type']}"


class ScriptCheck(Check):
    """
    Gate a candidate with an external script and private snapshotted files.

    Script failures fail closed; executor exceptions propagate. Construction
    validates settings without reading files or importing the script.

    Parameters
    ----------
    config : ScriptCheck.Config or None, optional
        Validated settings, mutually exclusive with keyword fields.
    **fields : Any
        Settings used to construct ``Config`` when no config is supplied.

    Examples
    --------
    >>> from swarm_arena import ScriptCheck
    >>> check = ScriptCheck(command=['python', 'checks/check.py'], feedback='pass_fail')
    >>> check.feedback
    'pass_fail'
    """

    name: ClassVar[str] = "script-check"
    _human_only: ClassVar[frozenset[str]] = frozenset({"command", "python", "files", "env"})

    class Config(_ConfigInputModel):
        """
        Configure a script check and its execution limits.

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
        feedback : {'pass_fail', 'reasons'}, default='reasons'
            Feedback exposed to agents.
        timeout_s : float or None, default=None
            Positive execution timeout, capped by the context's limit.
        grace_s : float, default=10.0
            Nonnegative shutdown grace period in seconds.
        resources : Resources, default=Resources()
            Executor scheduling and resource request.

        Examples
        --------
        >>> from swarm_arena import ScriptCheck
        >>> ScriptCheck.Config(command=['x']).feedback
        'reasons'
        """

        model_config = ConfigDict(extra="forbid", frozen=True)

        command: Annotated[list[str], Field(min_length=1)]
        python: str | None = None
        files: FileSet = FileSet()
        env: list[Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]] = []
        feedback: Literal["pass_fail", "reasons"] = "reasons"
        timeout_s: Annotated[float, Field(gt=0)] | None = None
        grace_s: Annotated[float, Field(ge=0)] = 10.0
        resources: Resources = Resources()

    def __init__(self, config: Config | None = None, /, **fields: Any) -> None:
        """Validate configuration without performing I/O."""
        super().__init__(config_or_fields(config, fields))
        self.feedback = self.config.feedback

    @property
    def config(self) -> Config:
        """
        Return the validated script-check settings.

        Returns
        -------
        ScriptCheck.Config
            Frozen configuration supplied at construction.

        Examples
        --------
        >>> from swarm_arena import ScriptCheck
        >>> ScriptCheck(command=['x']).config.command
        ['x']
        """
        return cast(ScriptCheck.Config, super().config)

    def preflight(self) -> None:
        """
        Require the check's files to be snapshotted.

        Raises
        ------
        RuntimeError
            If source globs have no manifest yet.

        Examples
        --------
        >>> from swarm_arena import ScriptCheck
        >>> ScriptCheck(command=['x']).preflight()
        """
        _preflight_files(self.name, self.config.files)

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        """
        Execute the check and fail closed on script or result errors.

        Parameters
        ----------
        candidate : Candidate
            Candidate exposed under ``candidate/`` in the script workspace.
        ctx : CheckContext
            Execution context providing goal files and resource scheduling.

        Returns
        -------
        CheckResult
            Strictly validated result or a failed check with a safe diagnostic.

        Raises
        ------
        Exception
            If the executor raises; lifecycle failures are never check results.

        Examples
        --------
        >>> import asyncio
        >>> from swarm_arena import Goal, Resources, ScriptCheck
        >>> from swarm_arena.board import Candidate
        >>> from swarm_arena.executor import Executor, Job
        >>> from swarm_arena.scoring import CheckContext
        >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
        >>> class MemoryExecutor(Executor):
        ...     async def submit(self, job: Job) -> ToolOutput:
        ...         return ToolOutput(status=RunStatus.OK, data={'passed': 'yes'})
        ...     async def capacity(self) -> Resources | None:
        ...         return None
        ...     async def aclose(self) -> None:
        ...         return None
        >>> goal = Goal(id='g', namespace='example')
        >>> candidate = Candidate(id='c1', run_id='r', goal_id='g', team='t',
        ...                       agent='a', round=1, request=ToolRequest(),
        ...                       output=ToolOutput(status=RunStatus.OK))
        >>> context = CheckContext(goal=goal, executor=MemoryExecutor(),
        ...                          artifact_root='/artifacts')
        >>> check = ScriptCheck(command=['example'])
        >>> result = asyncio.run(check.check(candidate, context))
        >>> result.passed
        False
        >>> result.reasons
        ['invalid check result: passed: bool_type']
        """
        out = await ctx.execute(
            script_command(self.config),
            files=self.config.files.manifest,
            candidate=candidate,
            resources=self.config.resources,
        )
        if out.status != RunStatus.OK:
            return CheckResult(
                check=self.name, passed=False, reasons=[f"check script failed: {out.status}"]
            )
        try:
            result = _CheckResultFile.model_validate(out.data)
        except ValidationError as error:
            return CheckResult(
                check=self.name,
                passed=False,
                reasons=[_result_error("check", _CheckResultFile, error)],
            )
        return CheckResult(
            check=self.name, passed=result.passed, reasons=result.reasons, details=result.details
        )
