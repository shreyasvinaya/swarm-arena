"""Run user scripts and expose read-only views of goal and candidate files."""

import warnings
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Annotated, Any, ClassVar, Protocol, cast

import anyio
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
)

from swarm_arena._errors import ConfigError, SoftError
from swarm_arena._model.base import JsonValue
from swarm_arena._model.goal import FileSet, Goal
from swarm_arena._model.ref import _ConfigInputModel, config_or_fields
from swarm_arena._model.results import Candidate
from swarm_arena._model.schema import _SchemaError, compile_object_schema
from swarm_arena._model.tooling import (
    DEFAULT_GRACE_S,
    Command,
    CompletedRun,
    EnvName,
    GraceS,
    TimeoutS,
    ToolOutput,
    ToolRequest,
)
from swarm_arena._tooling.paths import check_relpath, fold_key
from swarm_arena._tooling.result_file import tool_output_from_result
from swarm_arena._tooling.shadowing import _can_create_python
from swarm_arena._tooling.staging import effective_files
from swarm_arena._tooling.tool import Action, Tool


class _EmptyInputs(_ConfigInputModel):
    """Provide an empty placeholder; use ScriptTool.get_input_model for inputs."""


class FileViewContext(Protocol):
    """Describe the agent's permitted file view without depending on its runtime."""

    @property
    def goal(self) -> Goal:
        """Return the agent's goal."""
        ...

    def get_candidate(self, id: str) -> Candidate:
        """Return a candidate in the agent's view."""
        ...

    def _read_artifact(self, sha256: str) -> bytes:
        """Read artifact bytes within the agent's permitted view."""
        ...


class _ScriptConfig(Protocol):
    """Describe execution settings shared by script tools, checks, and judges."""

    @property
    def command(self) -> list[str]:
        """Return the argument vector."""
        ...

    @property
    def python(self) -> str | None:
        """Return the optional Python interpreter."""
        ...

    @property
    def env(self) -> list[str]:
        """Return the environment allowlist."""
        ...

    @property
    def timeout_s(self) -> float | None:
        """Return the configured timeout."""
        ...

    @property
    def grace_s(self) -> float:
        """Return the shutdown grace period."""
        ...


def script_command(config: _ScriptConfig) -> Command:
    """Build a command from shared script-component settings.

    Parameters
    ----------
    config : _ScriptConfig
        Argument vector, interpreter, environment allowlist, and time limits.

    Returns
    -------
    Command
        Command retaining the configured argument spelling and execution settings.

    Examples
    --------
    >>> from swarm_arena import ScriptTool
    >>> script_command(ScriptTool.Config(command=["python", "run.py"])).argv
    ['python', 'run.py']
    """
    return Command(
        argv=config.command,
        python=config.python,
        env=config.env,
        timeout_s=config.timeout_s,
        grace_s=config.grace_s,
    )


class ListFilesArgs(_ConfigInputModel):
    """Select goal files or a candidate's effective files."""

    candidate: str | None = None


class ReadFileArgs(_ConfigInputModel):
    """Select one path in the goal or a candidate's effective files."""

    path: str
    candidate: str | None = None


def _files(ctx: FileViewContext, candidate: str | None) -> dict[str, str]:
    """Resolve only files visible through the supplied agent context."""
    return (
        ctx.goal.files.manifest
        if candidate is None
        else effective_files(ctx.goal, ctx.get_candidate(candidate))
    )


def _list_files_sync(ctx: FileViewContext, args: ListFilesArgs) -> JsonValue:
    """List paths without fetching their artifact contents."""
    paths: list[JsonValue] = list(sorted(_files(ctx, args.candidate)))
    return {"files": paths}


async def _list_files(ctx: FileViewContext, args: ListFilesArgs) -> JsonValue:
    """Resolve a file listing without blocking the event loop."""
    return await anyio.to_thread.run_sync(_list_files_sync, ctx, args)


def _read_file_sync(ctx: FileViewContext, args: ReadFileArgs) -> JsonValue:
    """Read strict UTF-8 text only after resolving a path in the agent's view."""
    files = _files(ctx, args.candidate)
    if args.path not in files:
        raise SoftError(f"no such file: {args.path}")
    data = ctx._read_artifact(files[args.path])
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        raise SoftError(f"not UTF-8 text: {args.path}") from None
    return {"path": args.path, "content": content}


async def _read_file(ctx: FileViewContext, args: ReadFileArgs) -> JsonValue:
    """Read artifact contents on a worker thread."""
    return await anyio.to_thread.run_sync(_read_file_sync, ctx, args)


def _as_relpath(argument: str) -> str | None:
    """Normalize a workspace-relative command path, ignoring other arguments."""
    try:
        return check_relpath(argument)
    except ValueError:
        return None


class ScriptTool(Tool):
    """
    Run a human-configured script on goal files and permitted edits.

    Construction performs no I/O. Command paths found in the root manifest and
    tool-owned files remain protected regardless of editable globs. Use
    ``get_input_model()`` for the schema compiled from this instance's ``params``;
    the class-level ``input_model`` is an empty, frozen placeholder.

    Parameters
    ----------
    config : ScriptTool.Config or None, optional
        Validated settings, mutually exclusive with keyword fields.
    **fields : Any
        Settings passed to ``Config`` when no explicit configuration is supplied.

    Examples
    --------
    >>> from swarm_arena import ScriptTool
    >>> tool = ScriptTool(command=["python", "problem/run.py"],
    ...                   editable=["problem/*.txt"],
    ...                   params={"epochs": {"type": "integer", "maximum": 400}})
    >>> tool.get_input_model()(epochs=20).model_dump()
    {'epochs': 20}
    """

    name: ClassVar[str] = "script"
    description: ClassVar[str] = "Runs your script on your files."
    input_model: ClassVar[type[BaseModel]] = _EmptyInputs
    _human_only: ClassVar[frozenset[str]] = frozenset(
        {"command", "python", "files", "env", "editable"}
    )

    class Config(_ConfigInputModel):
        """
        Configure a script command, file permissions, and input parameters.

        Validation errors expose field names and error codes only; nested
        locations and rejected values are hidden.

        Parameters
        ----------
        command : list[str]
            Argument vector executed in the staged workspace.
        python : str or None, default=None
            Interpreter replacing a leading ``python`` argument.
        files : FileSet, default=FileSet()
            Tool-owned protected files; a list of globs is accepted as shorthand.
        editable : list[str], default=[]
            Globs for files the manager may edit, excluding protected files.
        params : dict[str, dict[str, Any]], default={}
            Supported JSON-schema properties for the script's parameters.
        timeout_s : float or None, default=None
            Positive execution timeout in seconds, if set.
        grace_s : float, default=10.0
            Nonnegative shutdown grace period in seconds.
        env : list[str], default=[]
            Environment variable names to allow into the process.
        deterministic : bool, default=True
            Whether repeated requests are expected to produce identical results.
        providers : list[str], default=[]
            Language-model providers used by the script.
        description : str, default=""
            Manager-facing description; empty selects the class description.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> ScriptTool.Config(command=["python", "run.py"]).grace_s
        10.0
        """

        model_config = ConfigDict(extra="forbid", frozen=True)

        command: Annotated[list[str], Field(min_length=1)]
        python: str | None = None
        files: FileSet = FileSet()
        editable: list[str] = []
        params: dict[str, dict[str, Any]] = {}
        timeout_s: TimeoutS | None = None
        grace_s: GraceS = DEFAULT_GRACE_S
        env: list[EnvName] = []
        deterministic: bool = True
        providers: list[str] = []
        description: str = ""

    def __init__(self, config: Config | None = None, /, **fields: Any) -> None:
        """Validate configuration without inspecting or importing the script."""
        super().__init__(config_or_fields(config, fields))
        self._input_model: type[BaseModel] | None = None
        for argument in self.config.command[1:]:
            script = _as_relpath(argument)
            if script is None or not fold_key(script).endswith(".py"):
                continue
            parent = Path(script).parent.as_posix()
            protected = {*self.config.files.manifest, script}
            if any(
                _can_create_python(pattern, parent, protected) for pattern in self.config.editable
            ):
                warnings.warn(
                    "editable glob permits Python import shadowing next to a protected script",
                    UserWarning,
                    stacklevel=2,
                )
                break

    @property
    def config(self) -> Config:
        """
        Return the validated script settings.

        Returns
        -------
        ScriptTool.Config
            Frozen configuration supplied at construction.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> ScriptTool(command=["x"]).config.command
        ['x']
        """
        return cast(ScriptTool.Config, super().config)

    def get_input_model(self) -> type[BaseModel]:
        """
        Compile and cache this instance's script parameter model.

        Input errors keep schema field names, aliases and list indices with
        built-in error codes; data-derived mapping and Any segments are hidden,
        as are rejected values.
        Schema errors retain property paths and keyword names, hiding configured
        values and suppressing original exception chains.

        Returns
        -------
        type[BaseModel]
            Strict, frozen model for the configured JSON-schema properties.

        Raises
        ------
        ConfigError
            If a parameter schema is invalid; values are omitted.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> ScriptTool(command=["x"]).get_input_model()().model_dump()
        {}
        """
        if self._input_model is None:
            name = f"{self.name}_params"
            try:
                compiled = compile_object_schema(name, self.config.params)
            except _SchemaError as error:
                raise ConfigError(str(error)) from None
            self._input_model = create_model(name, __base__=(compiled, _ConfigInputModel))
        return self._input_model

    def get_description(self) -> str:
        """
        Return the configured description or the class default.

        Returns
        -------
        str
            Description shown to the manager.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> ScriptTool(command=["x"], description="Tune it.").get_description()
        'Tune it.'
        """
        return self.config.description or self.description

    def editable(self) -> Sequence[str]:
        """
        Return the configured editable globs.

        Returns
        -------
        Sequence[str]
            Patterns permitting edits except at protected paths.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> tuple(ScriptTool(command=["x"], editable=["problem/*.txt"]).editable())
        ('problem/*.txt',)
        """
        return tuple(self.config.editable)

    def is_deterministic(self, request: ToolRequest) -> bool:
        """
        Report whether the script is configured as deterministic.

        Parameters
        ----------
        request : ToolRequest
            Invocation whose repeat behavior is being assessed.

        Returns
        -------
        bool
            Configured determinism flag.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> from swarm_arena.tool import ToolRequest
        >>> ScriptTool(command=["x"]).is_deterministic(ToolRequest())
        True
        """
        return self.config.deterministic

    def providers(self) -> frozenset[str]:
        """
        Return the providers declared by the script configuration.

        Returns
        -------
        frozenset[str]
            Unique language-model provider names.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> sorted(ScriptTool(command=["x"], providers=["anthropic"]).providers())
        ['anthropic']
        """
        return frozenset(self.config.providers)

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        """
        Build the configured command for the staged workspace.

        Parameters
        ----------
        request : ToolRequest
            Invocation whose parameters are staged in ``swarm/request.json``.
        workspace : Path
            Executor-owned working directory.

        Returns
        -------
        Command
            Argument vector, interpreter, environment allowlist, and time limits.

        Examples
        --------
        >>> from pathlib import Path
        >>> from swarm_arena import ScriptTool
        >>> from swarm_arena.tool import ToolRequest
        >>> ScriptTool(command=["python", "run.py"]).command(ToolRequest(), Path("ws")).argv
        ['python', 'run.py']
        """
        return script_command(self.config)

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        """
        Collect the script's result using the shared result-file rules.

        Parameters
        ----------
        request : ToolRequest
            Invocation originally passed to ``command``.
        workspace : Path
            Workspace containing ``swarm/result.json`` and declared outputs.
        run : CompletedRun
            Process status, timestamps, and log paths.

        Returns
        -------
        ToolOutput
            Validated result, retaining partial data on timeout or cancellation.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> output = ScriptTool(command=["x"]).collect(request, workspace, run)  # doctest: +SKIP
        """
        return tool_output_from_result(workspace, run)

    def extra_actions(self) -> tuple[Action[ListFilesArgs], Action[ReadFileArgs]]:
        """
        Return read-only file actions charged as board reads.

        Returns
        -------
        tuple[Action[ListFilesArgs], Action[ReadFileArgs]]
            List and read actions restricted to the context's file view.

        Examples
        --------
        >>> from swarm_arena import ScriptTool
        >>> [action.name for action in ScriptTool(command=["x"]).extra_actions()]
        ['list_files', 'read_file']
        """
        return (
            Action(
                name="list_files",
                description="List goal files or a candidate's effective files.",
                input_model=ListFilesArgs,
                handler=_list_files,
                counts_as="board_read",
            ),
            Action(
                name="read_file",
                description="Read a UTF-8 file from the goal or a candidate's effective files.",
                input_model=ReadFileArgs,
                handler=_read_file,
                counts_as="board_read",
            ),
        )

    def _own_files(self) -> dict[str, str]:
        """Return the tool's protected file manifest."""
        return self.config.files.manifest

    def _protected(self, root: AbstractSet[str]) -> frozenset[str]:
        """Match folded command paths, retaining root spellings and all own files."""
        command_keys = {
            fold_key(path)
            for argument in (*self.config.command, self.config.python)
            if argument and (path := _as_relpath(argument)) is not None
        }
        return frozenset(path for path in root if fold_key(path) in command_keys) | frozenset(
            self.config.files.manifest
        )
