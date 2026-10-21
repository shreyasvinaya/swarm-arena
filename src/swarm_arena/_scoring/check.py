"""Define candidate checks and the context for executing scoring commands."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from typing import ClassVar, Literal

from swarm_arena._exec.env import tool_env_vars
from swarm_arena._exec.executor import Executor, Job
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.goal import Goal
from swarm_arena._model.ids import IdGen
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Candidate, CheckResult
from swarm_arena._model.tooling import Command, ToolOutput
from swarm_arena._outcomes import _propagate_control
from swarm_arena._tooling.staging import effective_files, plan_script_stage
from swarm_arena._tooling.tool import cap_timeout


def _prepare_command_output(value: object) -> ToolOutput:
    """Fail closed on invalid executor models without exposing subclass operations."""
    try:
        return _detach_model(ToolOutput, value)
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        _propagate_control(error)
        return ToolOutput(status="error")


class CheckContext:
    """
    Execute check commands with run identity and candidate staging.

    Parameters
    ----------
    goal : Goal
        Goal whose effective files are exposed to a check.
    executor : Executor
        Executor receiving command jobs.
    artifact_root : str
        Root holding content-addressed workspace artifacts.
    run_id : str, default='local'
        Run identity for environment variables and default ids.
    round : int, default=0
        Current round number.
    part : str or None, default=None
        Part being checked, if any.
    on_started : Callable[[Job.Started], Awaitable[None]] or None, default=None
        Callback placed on each command job before execution.
    secrets_file : str or None, default=None
        Executor-local file resolving allowlisted environment names.
    ids : IdGen or None, default=None
        Shared run id source; None creates ``IdGen(run_id, 0)``.
    tool_timeout_s : float, default=3600.0
        Human tool-time limit applied to every command.

    Examples
    --------
    >>> import asyncio
    >>> from swarm_arena import Goal, Resources
    >>> from swarm_arena.executor import Executor, Job
    >>> from swarm_arena.scoring import CheckContext
    >>> from swarm_arena.tool import Command, RunStatus, ToolOutput
    >>> class RecordingExecutor(Executor):
    ...     def __init__(self) -> None:
    ...         self.jobs: list[Job] = []
    ...     async def submit(self, job: Job) -> ToolOutput:
    ...         self.jobs.append(job)
    ...         return ToolOutput(status=RunStatus.OK)
    ...     async def capacity(self) -> Resources | None:
    ...         return None
    ...     async def aclose(self) -> None:
    ...         return None
    >>> executor = RecordingExecutor()
    >>> context = CheckContext(goal=Goal(id='g', namespace='example'),
    ...                        executor=executor, artifact_root='/artifacts')
    >>> output = asyncio.run(context.execute(Command(argv=['example'])))
    >>> output.status == RunStatus.OK
    True
    >>> executor.jobs[0].command.argv
    ['example']
    """

    def __init__(
        self,
        *,
        goal: Goal,
        executor: Executor,
        artifact_root: str,
        run_id: str = "local",
        round: int = 0,
        part: str | None = None,
        on_started: Callable[[Job.Started], Awaitable[None]] | None = None,
        secrets_file: str | None = None,
        ids: IdGen | None = None,
        tool_timeout_s: float = 3600.0,
    ) -> None:
        """Retain execution settings without performing I/O."""
        self.goal = goal
        self.executor = executor
        self.artifact_root = artifact_root
        self.run_id = run_id
        self.round = round
        self.part = part
        self.on_started = on_started
        self.secrets_file = secrets_file
        self.ids = IdGen(run_id, 0) if ids is None else ids
        self.tool_timeout_s = tool_timeout_s

    async def execute(
        self,
        command: Command,
        *,
        files: Mapping[str, str] | None = None,
        candidate: Candidate | None = None,
        resources: Resources | None = None,
    ) -> ToolOutput:
        """
        Submit a command with own files and optional read-only candidate data.

        Executor exceptions propagate to the caller, including fencing failures
        from ``on_started``. Process failures are represented by output status.

        Parameters
        ----------
        command : Command
            Program and execution settings.
        files : Mapping[str, str] or None, default=None
            Own workspace paths mapped to artifact hashes.
        candidate : Candidate or None, default=None
            Candidate staged under ``candidate/`` with effective and output files.
        resources : Resources or None, default=None
            Scheduling request; None selects ``Resources()``.

        Returns
        -------
        ToolOutput
            Detached executor result with parsed ``swarm/result.json`` data.
            Invalid executor outputs become ``ToolOutput(status="error")``.

        Examples
        --------
        The :class:`CheckContext` example uses ``asyncio.run(context.execute(...))``
        with an in-memory recording executor and inspects the submitted command.
        """
        stage = (
            plan_script_stage(
                own_files=files or {},
                candidate=candidate,
                effective_files=effective_files(self.goal, candidate),
            )
            if candidate is not None
            else Job.Stage(root=dict(files or {}))
        )
        output = await self.executor.submit(
            Job(
                id=self.ids.next("x"),
                kind="command",
                command=command,
                stage=stage,
                timeout_s=cap_timeout(command.timeout_s, self.tool_timeout_s),
                grace_s=command.grace_s,
                resources=resources or Resources(),
                env=tool_env_vars(
                    run_id=self.run_id,
                    goal_id=self.goal.id,
                    candidate_id=candidate.id if candidate is not None else "",
                    team="",
                    agent="",
                    round=self.round,
                ),
                secret_env=tuple(command.env),
                secrets_file=self.secrets_file,
                artifact_root=self.artifact_root,
                on_started=self.on_started,
            )
        )
        return await run_sync_daemon(_prepare_command_output, output)


class Check(Configurable, ABC):
    """
    Gate a candidate using an asynchronous validation check.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None, optional
        Settings validated by the subclass's nested ``Config`` model.
        None selects that model's defaults.

    Attributes
    ----------
    name : ClassVar[str]
        Check name.
    feedback : {'pass_fail', 'reasons'}
        Feedback exposed to agents; defaults to reasons.

    Examples
    --------
    >>> import asyncio
    >>> from unittest.mock import Mock
    >>> from swarm_arena.board import Candidate
    >>> from swarm_arena.scoring import Check, CheckContext, CheckResult
    >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
    >>> sorted(Check.__abstractmethods__)
    ['check']
    >>> class ExampleCheck(Check):
    ...     name = 'example'
    ...     async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
    ...         return CheckResult(check=self.name, passed=not candidate.gated_out)
    >>> check = ExampleCheck()
    >>> check.preflight()
    >>> candidate = Candidate(id='c1', run_id='r', goal_id='g', team='t', agent='a',
    ...     round=1, request=ToolRequest(), output=ToolOutput(status=RunStatus.OK))
    >>> result = asyncio.run(check.check(candidate, Mock(spec=CheckContext)))
    >>> (result.check, result.passed, check.feedback)
    ('example', True, 'reasons')
    """

    name: ClassVar[str]
    feedback: Literal["pass_fail", "reasons"] = "reasons"

    def preflight(self) -> None:
        """
        Check readiness before evaluating candidates.

        The default implementation performs no I/O or validation.

        Examples
        --------
        The :class:`Check` example calls ``check.preflight()`` without I/O.
        """

    @abstractmethod
    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        """
        Evaluate one candidate and return its gating result.

        Parameters
        ----------
        candidate : Candidate
            Candidate to validate.
        ctx : CheckContext
            Goal and command execution context.

        Returns
        -------
        CheckResult
            Pass or fail result and any feedback.

        Examples
        --------
        The :class:`Check` example runs ``check.check(candidate, context)`` with
        ``asyncio.run`` and inspects the result entirely in memory.
        """
