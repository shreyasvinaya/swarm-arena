"""Define staged jobs and the asynchronous executor contract."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from pydantic import field_validator

from swarm_arena._model.base import FrozenModel, StoredModel
from swarm_arena._model.plan import Resources
from swarm_arena._model.ref import Configurable, Ref
from swarm_arena._model.tooling import Command, ParentRef, ToolOutput, ToolRequest
from swarm_arena._tooling.paths import check_key


@dataclass(frozen=True, kw_only=True)
class Job:
    """
    Describe a staged tool invocation or command for an executor.

    Callbacks are local control hooks and must be omitted from serialized job
    payloads. Secret values travel only in the task payload and must never be
    journaled or logged. Neither callbacks nor secret values appear in repr.

    Parameters
    ----------
    id : str
        Candidate id or command-job id.
    kind : {'tool', 'command'}
        Select a tool reference with a request, or an explicit command.
    tool : Ref or None, default=None
        Tool to build on the executor for a tool job.
    command : Command or None, default=None
        Command to execute for a command job.
    request : ToolRequest or None, default=None
        Invocation supplied to the tool.
    stage : Job.Stage
        Driver-planned workspace contents.
    timeout_s : float or None
        Execution timeout in seconds, or None for no timeout.
    grace_s : float
        Grace period in seconds before forced termination.
    resources : Resources
        Resource request used for scheduling.
    env : Mapping[str, str]
        Nonsecret environment values supplied by the driver.
    secret_env : tuple[str, ...]
        Environment variable names resolved on the executor's node.
    secrets_file : str or None
        Optional file used to resolve allowed secret names.
    artifact_root : str
        Root holding content-addressed workspace artifacts and logs.
    secret_values : Mapping[str, str], optional
        Driver-supplied secrets overriding other environment sources.
    on_started : Callable[[Job.Started], Awaitable[None]] or None, default=None
        Await before releasing the job to run; None starts without a callback.
    on_partial : Callable[[ToolOutput], Awaitable[None]] or None, default=None
        Receive partial output inside the shielded stop sequence when submit
        is cancelled. This is the only cancel-path partial-output channel.

    Examples
    --------
    >>> from swarm_arena import Resources
    >>> from swarm_arena.executor import Job
    >>> from swarm_arena.tool import Command
    >>> job = Job(id="x-1", kind="command", command=Command(argv=["true"]),
    ...           stage=Job.Stage(), timeout_s=5.0, grace_s=1.0,
    ...           resources=Resources(), env={}, secret_env=(),
    ...           secrets_file=None, artifact_root="/tmp/a")
    >>> job.kind
    'command'
    """

    class Stage(FrozenModel):
        """
        Describe workspace contents computed in the driver.

        Validation keeps schema field names, aliases and list/tuple indices;
        unknown model keys become ``<field>`` and dynamic data segments ``<…>``.
        Stage owns this hidden-key policy throughout its validated tree.

        Parameters
        ----------
        root : dict[str, str], optional
            Workspace paths in normal form mapped to SHA-256 artifact hashes.
        parents : list[ParentRef], optional
            Parents staged read-only under ``parents/<id>/``.
        candidate : Job.CandidateFiles or None, default=None
            Candidate data staged for a script check or judge.
        request_json : dict[str, Any] or None, default=None
            Request document written to ``swarm/request.json``.

        Examples
        --------
        >>> from swarm_arena.executor import Job
        >>> Job.Stage(root={"problem/input.txt": "a" * 64}).parents
        []
        """

        root: dict[str, str] = {}
        parents: list[ParentRef] = []
        candidate: "Job.CandidateFiles | None" = None
        request_json: dict[str, Any] | None = None

        @field_validator("root")
        @classmethod
        def _check_root(cls, root: dict[str, str]) -> dict[str, str]:
            """Require all root paths to already be in normal form."""
            for path in root:
                check_key(path)
            return root

    class CandidateFiles(FrozenModel):
        """
        Describe the candidate directory for a script check or judge.

        Parameters
        ----------
        request : dict[str, Any]
            Candidate request document.
        output : dict[str, Any]
            Candidate output document.
        files : dict[str, str]
            Effective and output paths in normal form mapped to artifact hashes.

        Examples
        --------
        >>> from swarm_arena.executor import Job
        >>> Job.CandidateFiles(request={}, output={}, files={}).files
        {}
        """

        request: dict[str, Any]
        output: dict[str, Any]
        files: dict[str, str]

        @field_validator("files")
        @classmethod
        def _check_files(cls, files: dict[str, str]) -> dict[str, str]:
            """Require all candidate file paths to already be in normal form."""
            for path in files:
                check_key(path)
            return files

    class Started(StoredModel):
        """
        Record a committed process start or remote dispatch.

        Parameters
        ----------
        schema_version : int, default=1
            Stored-format version upgraded through the migration registry.
        kind : {'process', 'dispatch'}
            Local process start or remote dispatch.
        job_id : str
            Candidate or command-job id.
        host_id : str or None, default=None
            Host identity used to verify ownership before orphan cleanup.
        pid : int or None, default=None
            Process id of the launcher.
        pgid : int or None, default=None
            Process group id of the job.
        start_time : str or None, default=None
            Recorded process start time used to guard against pid reuse.
        task_name : str or None, default=None
            Remote task name for a dispatch.

        Examples
        --------
        >>> from swarm_arena.executor import Job
        >>> started = Job.Started(kind="process", job_id="c-1")
        >>> Job.Started.model_validate_json(started.model_dump_json()) == started
        True
        """

        schema_kind: ClassVar[str] = "job_started"

        kind: Literal["process", "dispatch"]
        job_id: str
        host_id: str | None = None
        pid: int | None = None
        pgid: int | None = None
        start_time: str | None = None
        task_name: str | None = None

    id: str
    kind: Literal["tool", "command"]
    tool: Ref | None = None
    command: Command | None = None
    request: ToolRequest | None = None
    stage: Stage
    timeout_s: float | None
    grace_s: float
    resources: Resources
    env: Mapping[str, str]
    secret_env: tuple[str, ...]
    secrets_file: str | None
    artifact_root: str
    secret_values: Mapping[str, str] = field(default_factory=dict, repr=False)
    on_started: Callable[[Started], Awaitable[None]] | None = field(default=None, repr=False)
    on_partial: Callable[[ToolOutput], Awaitable[None]] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Require the payload selected by the job kind."""
        if self.kind == "tool" and (self.tool is None or self.request is None):
            raise ValueError("kind 'tool' needs tool and request")
        if self.kind == "command" and self.command is None:
            raise ValueError("kind 'command' needs command")


Job.Stage.model_rebuild()


class Executor(Configurable, ABC):
    """
    Submit jobs and manage execution resources asynchronously.

    Construction validates configuration without I/O or referenced-code imports.
    Subclasses implement submission, capacity reporting, and cleanup.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None, optional
        Settings validated by the subclass's nested ``Config`` model.
        None selects that model's defaults.

    Examples
    --------
    >>> from swarm_arena.executor import Executor
    >>> sorted(Executor.__abstractmethods__)
    ['aclose', 'capacity', 'submit']
    """

    def preflight(self) -> None:
        """
        Check readiness before submitting jobs.

        The default implementation performs no checks or I/O.

        Examples
        --------
        >>> executor.preflight()  # doctest: +SKIP
        """

    @abstractmethod
    async def submit(self, job: Job) -> ToolOutput:
        """
        Run a job and return its completed output.

        Cancelling the await must stop the job in a shielded sequence: signal
        termination, allow grace, collect partial output, and force termination.
        Deliver cancel-path partial output through ``job.on_partial`` inside that
        sequence. Await ``job.on_started`` before allowing the job to run.

        Parameters
        ----------
        job : Job
            Staged invocation, limits, environment, and lifecycle callbacks.

        Returns
        -------
        ToolOutput
            Execution result, including any timeout output.

        Examples
        --------
        >>> output = await executor.submit(job)  # doctest: +SKIP
        """

    @abstractmethod
    async def capacity(self) -> Resources | None:
        """
        Return the executor's available resource capacity.

        Returns
        -------
        Resources or None
            Resource capacity, or None when unknown.

        Examples
        --------
        >>> capacity = await executor.capacity()  # doctest: +SKIP
        """

    @abstractmethod
    async def aclose(self) -> None:
        """
        Release resources held by the executor.

        Examples
        --------
        >>> await executor.aclose()  # doctest: +SKIP
        """

    async def confirm_dead(self, started: Job.Started) -> bool | None:
        """
        Report whether a recorded job is known to have stopped.

        Parameters
        ----------
        started : Job.Started
            Committed start identity to inspect.

        Returns
        -------
        bool or None
            True if dead, False if alive, or None when unknown. The default
            cannot determine liveness and returns None.

        Examples
        --------
        >>> dead = await executor.confirm_dead(started)  # doctest: +SKIP
        """
        return None

    async def kill_orphan(self, started: Job.Started) -> bool:
        """
        Attempt to terminate an orphan identified by its committed start.

        Parameters
        ----------
        started : Job.Started
            Start identity whose host and process start time must be verified.

        Returns
        -------
        bool
            Whether orphan termination succeeded. The default takes no action
            and returns False.

        Examples
        --------
        >>> killed = await executor.kill_orphan(started)  # doctest: +SKIP
        """
        return False
