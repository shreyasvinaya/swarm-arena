"""Wrap annotated Python functions as subprocess tools."""

import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, cast, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    create_model,
)

from swarm_arena._model.ref import (
    _ConfigInputModel,
    config_or_fields,
    import_path,
    load_object,
)
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
from swarm_arena._outcomes import _classify_outcome
from swarm_arena._tooling.result_file import tool_output_from_result
from swarm_arena._tooling.tool import Tool


class _EmptyInputs(_ConfigInputModel):
    """Provide a frozen placeholder; use FunctionTool.get_input_model for inputs."""


def _load_function(path: str) -> Callable[..., Any]:
    """Load through the shared cache without exposing import exception values."""
    module_name = path.split(":", 1)[0]
    fn = load_object(path)
    module = sys.modules.get(module_name)
    # sys.modules can hold any object, so the spec is read defensively.
    spec = getattr(module, "__spec__", None)
    real = getattr(spec, "name", None)
    if spec is None or real != module_name:
        alias = f"is an alias of {real}" if isinstance(real, str) else "has no import spec"
        raise TypeError(f"FunctionTool {path}: {module_name} {alias}; use the defining module")
    if not callable(fn):
        raise TypeError(f"FunctionTool {path}: expected a callable function")
    return fn


def function_input_model(fn: Callable[..., Any]) -> type[BaseModel]:
    """
    Build a parameter model from an importable function's annotated signature.

    Parameters
    ----------
    fn : Callable[..., Any]
        Synchronous function with annotated positional-or-keyword or keyword-only
        parameters. Variadic and positional-only parameters are unsupported.

    Returns
    -------
    type[BaseModel]
        Model named after the function, preserving defaults and forbidding extras.

    Raises
    ------
    TypeError
        If the function, signature, annotations, or parameter types are unsupported.

    Examples
    --------
    >>> model = function_input_model(import_root)
    >>> model(module_name="swarm_arena").model_dump()
    {'module_name': 'swarm_arena'}
    """
    path = import_path(fn)
    if not inspect.isfunction(fn):
        raise TypeError(f"FunctionTool {path}: expected a Python function")
    if inspect.iscoroutinefunction(fn):
        raise TypeError(f"FunctionTool {path}: coroutine functions are unsupported")
    try:
        signature = inspect.signature(fn)
        hints = get_type_hints(fn, include_extras=True)
    except BaseException as error:
        _classify_outcome(error)
        raise TypeError(f"FunctionTool {path}: cannot resolve signature or annotations") from None
    fields: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            raise TypeError(f"FunctionTool {path}: unsupported parameter kind for {name}")
        if name not in hints:
            raise TypeError(f"FunctionTool {path}: parameter {name} needs an annotation")
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[name] = (hints[name], default)
    try:
        return create_model(f"{fn.__name__}_inputs", __base__=_ConfigInputModel, **fields)
    except BaseException as error:
        _classify_outcome(error)
        raise TypeError(f"FunctionTool {path}: cannot build input model") from None


def import_root(module_name: str) -> str:
    """
    Find the import root of a module or package with a filesystem location.

    Parameters
    ----------
    module_name : str
        Dotted module name from the configured function import path.

    Returns
    -------
    str
        Resolved directory containing the loaded module's real top-level package.
        A package is known by its spec's submodule search locations, so an
        ``__init__`` that is a sourceless ``.pyc`` or a compiled extension works.
        FunctionTool rejects aliases of that module during preflight.

    Raises
    ------
    ValueError
        If the module cannot be imported or has no usable file location.

    Examples
    --------
    >>> Path(import_root("swarm_arena")).is_dir()
    True
    """
    load_object(module_name + ":__name__")
    try:
        module = sys.modules.get(module_name)
        spec = getattr(module, "__spec__", None)
        file = getattr(module, "__file__", None)
        if file is None or spec is None:
            raise ValueError
        # A package has submodule search locations, whatever its __init__ file is
        # (.py, a sourceless .pyc, or a compiled extension).
        levels = len(spec.name.split(".")) + (spec.submodule_search_locations is not None)
        return str(Path(file).resolve().parents[levels - 1])
    except BaseException as error:
        _classify_outcome(error)
        raise ValueError(f"cannot determine import root for {module_name}") from None


class FunctionTool(Tool):
    """
    Run an importable Python function in a supervised subprocess.

    Construction stores the reference without importing it. Use
    ``get_input_model()`` to inspect parameters; ``input_model`` is only an
    empty frozen placeholder. The function must return a finite JSON object.
    Files written by the function are never uploaded, including checkpoints
    retained after timeout or cancellation. Use ScriptTool or a custom Tool
    when output files are needed. Environment names are explicitly allowlisted
    with ``env`` and cannot be made planner-settable.
    The configured module must use its real import name; ``sys.modules`` aliases,
    and entries without an import spec (such as a class put there), are
    rejected during preflight with an identifier-only TypeError.

    Parameters
    ----------
    function : str or Callable[..., Any] or FunctionTool.Config
        Module-level function, its ``module:qualname`` path, or full configuration.
    **fields : Any
        Configuration fields when passing a function or path. Pass ``function``
        positionally only; a duplicate keyword raises TypeError.

    Examples
    --------
    >>> from swarm_arena import FunctionTool
    >>> tool = FunctionTool("my_package.tools:solve", timeout_s=30)
    >>> tool.config.function
    'my_package.tools:solve'
    """

    _human_only: ClassVar[frozenset[str]] = frozenset({"env"})
    name: ClassVar[str] = "function"
    description: ClassVar[str] = "Runs a Python function in a subprocess."
    input_model: ClassVar[type[BaseModel]] = _EmptyInputs
    """Default empty input schema; use ``get_input_model`` for the configured schema."""

    class Config(_ConfigInputModel):
        """
        Configure a function tool without importing the function.

        Parameters
        ----------
        function : str
            Import path in ``module:qualname`` form.
        deterministic : bool, default=True
            Whether repeated requests should yield the same result.
        providers : list[str], default=[]
            Language-model provider names used by the function.
        env : list[str], default=[]
            Human-configured names forwarded through Command.env.
        timeout_s : float or None, default=None
            Positive execution timeout, or None to use the run's tool timeout.
        grace_s : float, default=10.0
            Nonnegative shutdown grace period before forced termination.

        Examples
        --------
        >>> from swarm_arena import FunctionTool
        >>> FunctionTool.Config(function="my_package.tools:solve").grace_s
        10.0
        """

        model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
        function: str
        deterministic: bool = True
        providers: list[str] = []
        env: list[EnvName] = []
        timeout_s: TimeoutS | None = None
        grace_s: GraceS = DEFAULT_GRACE_S

    def __init__(
        self, function: "str | Callable[..., Any] | FunctionTool.Config", /, **fields: Any
    ) -> None:
        """Validate settings while retaining only the function's import path."""
        if "function" in fields:
            raise TypeError("pass function positionally only")
        if isinstance(function, self.Config):
            data = config_or_fields(function, fields)
        else:
            path = import_path(function) if callable(function) else function
            data = {**fields, "function": path}
        super().__init__(data)

    @property
    def config(self) -> Config:
        """
        Return the validated function configuration.

        Returns
        -------
        FunctionTool.Config
            Frozen settings containing the function's import path.

        Examples
        --------
        >>> from swarm_arena import FunctionTool
        >>> FunctionTool("my_package.tools:solve").config.deterministic
        True
        """
        return cast(FunctionTool.Config, super().config)

    def get_input_model(self) -> type[BaseModel]:
        """
        Import the function lazily and build its parameter model.

        Returns
        -------
        type[BaseModel]
            Model derived from the function's annotated signature.

        Examples
        --------
        >>> from swarm_arena import FunctionTool
        >>> FunctionTool("textwrap:dedent").get_input_model()
        Traceback (most recent call last):
        ...
        TypeError: FunctionTool textwrap:dedent: parameter text needs an annotation
        """
        return function_input_model(_load_function(self.config.function))

    def preflight(self) -> None:
        """
        Check that the referenced function has a supported input signature.

        Examples
        --------
        >>> from swarm_arena import FunctionTool
        >>> FunctionTool("textwrap:dedent").preflight()
        Traceback (most recent call last):
        ...
        TypeError: FunctionTool textwrap:dedent: parameter text needs an annotation
        """
        self.get_input_model()

    def get_description(self) -> str:
        """
        Return the function's first docstring line or the default description.

        Returns
        -------
        str
            Description shown to the manager.

        Examples
        --------
        >>> from swarm_arena import FunctionTool
        >>> FunctionTool("textwrap:dedent").get_description()
        'Remove any common leading whitespace from every line in `text`.'
        """
        doc = inspect.getdoc(_load_function(self.config.function))
        return doc.splitlines()[0] if doc else self.description

    def is_deterministic(self, request: ToolRequest) -> bool:
        """
        Return the configured determinism flag.

        Parameters
        ----------
        request : ToolRequest
            Invocation whose determinism is being assessed.

        Returns
        -------
        bool
            Whether repeat invocations should yield the same result.

        Examples
        --------
        >>> from swarm_arena import FunctionTool
        >>> from swarm_arena.tool import ToolRequest
        >>> FunctionTool("my_package.tools:solve").is_deterministic(ToolRequest())
        True
        """
        return self.config.deterministic

    def providers(self) -> frozenset[str]:
        """
        Return the configured language-model provider names.

        Returns
        -------
        frozenset[str]
            Providers the function may call.

        Examples
        --------
        >>> from swarm_arena import FunctionTool
        >>> FunctionTool("my_package.tools:solve", providers=["openai"]).providers()
        frozenset({'openai'})
        """
        return frozenset(self.config.providers)

    def command(self, request: ToolRequest, workspace: Path) -> Command:
        """
        Build the subprocess command with the function's explicit import root.

        Parameters
        ----------
        request : ToolRequest
            Invocation staged in ``swarm/request.json``.
        workspace : Path
            Staged workspace where the command will execute.

        Returns
        -------
        Command
            Python command using ``-P`` and the configured execution limits.

        Examples
        --------
        >>> from pathlib import Path
        >>> from swarm_arena import FunctionTool
        >>> from swarm_arena.tool import ToolRequest
        >>> cmd = FunctionTool("textwrap:dedent").command(ToolRequest(), Path("/ws"))
        >>> cmd.argv[:4]
        ['python', '-P', '-m', 'swarm_arena.run_function']
        """
        _load_function(self.config.function)
        return Command(
            argv=[
                "python",
                "-P",
                "-m",
                "swarm_arena.run_function",
                "--import-root",
                import_root(self.config.function.split(":", 1)[0]),
                self.config.function,
            ],
            env=self.config.env,
            timeout_s=self.config.timeout_s,
            grace_s=self.config.grace_s,
        )

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        """
        Collect structured function data, ignoring output-file declarations.

        Parameters
        ----------
        request : ToolRequest
            Invocation that produced the result.
        workspace : Path
            Workspace containing ``swarm/result.json``.
        run : CompletedRun
            Process status, timestamps, and logs.

        Returns
        -------
        ToolOutput
            Validated result and process metadata.

        Examples
        --------
        >>> from datetime import UTC, datetime
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> from swarm_arena import FunctionTool
        >>> from swarm_arena.tool import CompletedRun, RunStatus, ToolRequest
        >>> tool = FunctionTool("my_tools:add")
        >>> now = datetime(2026, 10, 18, tzinfo=UTC)
        >>> with TemporaryDirectory() as directory:
        ...     workspace = Path(directory)
        ...     (workspace / "swarm").mkdir()
        ...     _ = (workspace / "swarm/result.json").write_text(
        ...         '{"schema_version": 1, "data": {"sum": 5}}')
        ...     run = CompletedRun(returncode=0, status=RunStatus.OK,
        ...         stdout_path=workspace / "stdout", stderr_path=workspace / "stderr",
        ...         started_at=now, ended_at=now)
        ...     output = tool.collect(ToolRequest(), workspace, run)
        >>> output.data
        {'sum': 5}
        """
        return tool_output_from_result(workspace, run, include_files=False)
