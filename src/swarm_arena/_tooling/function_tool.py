"""Wrap annotated Python functions as subprocess tools."""

import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, LiteralString, cast, get_type_hints

from pydantic import (
    BaseModel,
    ConfigDict,
    ModelWrapValidatorHandler,
    ValidationError,
    create_model,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from swarm_arena._model.ref import (
    _CONFIG_ERROR_TYPES,
    _validate_config,
    config_or_fields,
    import_path,
    load_object,
)
from swarm_arena._model.tooling import Command, CompletedRun, ToolOutput, ToolRequest
from swarm_arena._tooling.result_file import tool_output_from_result
from swarm_arena._tooling.tool import Tool


def _sanitize_validation(
    cls: type[BaseModel], data: Any, handler: ModelWrapValidatorHandler[Any]
) -> Any:
    """Expose only top-level schema fields and built-in validation error codes."""
    try:
        return handler(data)
    except ValidationError as exc:
        errors: list[InitErrorDetails] = []
        for error in exc.errors(include_input=False, include_url=False, include_context=False):
            loc = error["loc"]
            first = loc[0] if loc else None
            field = first if isinstance(first, str) and first in cls.model_fields else "<field>"
            code = cast(
                LiteralString, error["type"] if error["type"] in _CONFIG_ERROR_TYPES else "invalid"
            )
            errors.append(
                InitErrorDetails(
                    type=PydanticCustomError(code, code),
                    loc=(field, "<…>") if len(loc) > 1 else (field,),
                    input=None,
                )
            )
        raise ValidationError.from_exception_data(cls.__name__, errors, hide_input=True) from None
    except Exception:
        raise ValidationError.from_exception_data(
            cls.__name__,
            [InitErrorDetails(type=PydanticCustomError("invalid", "invalid"), loc=(), input=None)],
            hide_input=True,
        ) from None


class _FunctionModel(BaseModel):
    """Hide values in configuration and generated-input validation failures."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    _sanitize_errors = model_validator(mode="wrap")(_sanitize_validation)


class _EmptyInputs(_FunctionModel):
    """Provide a frozen placeholder; use FunctionTool.get_input_model for inputs."""

    model_config = ConfigDict(frozen=True)


def _load_function(path: str) -> Callable[..., Any]:
    """Load through the shared cache without exposing import exception values."""
    try:
        fn = load_object(path)
    except Exception:
        raise ValueError(f"cannot import {path}") from None
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
    except Exception:
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
        return create_model(f"{fn.__name__}_inputs", __base__=_FunctionModel, **fields)
    except Exception:
        raise TypeError(f"FunctionTool {path}: cannot build input model") from None


def import_root(module_name: str) -> str:
    """
    Find the import root of a module or package with a filesystem location.

    Parameters
    ----------
    module_name : str
        Dotted module name, as recorded on the function.

    Returns
    -------
    str
        Resolved directory containing the module's top-level package.

    Raises
    ------
    ValueError
        If the module cannot be imported or has no usable file location.

    Examples
    --------
    >>> Path(import_root("swarm_arena")).is_dir()
    True
    """
    try:
        module = importlib.import_module(module_name)
        if module.__file__ is None:
            raise ValueError
        file = Path(module.__file__).resolve()
        levels = len(module_name.split(".")) + (file.name == "__init__.py")
        return str(file.parents[levels - 1])
    except Exception:
        raise ValueError(f"cannot determine import root for {module_name}") from None


class FunctionTool(Tool):
    """
    Run an importable Python function in a supervised subprocess.

    Construction stores the reference without importing it. Use
    ``get_input_model()`` to inspect parameters; ``input_model`` is only an
    empty frozen placeholder. The function must return a finite JSON object.

    Parameters
    ----------
    function : str or Callable[..., Any] or FunctionTool.Config
        Module-level function, its ``module:qualname`` path, or full configuration.
    **fields : Any
        Configuration fields when passing a function or path.

    Examples
    --------
    >>> from swarm_arena import FunctionTool
    >>> tool = FunctionTool("my_package.tools:solve", timeout_s=30)
    >>> tool.config.function
    'my_package.tools:solve'
    """

    name: ClassVar[str] = "function"
    description: ClassVar[str] = "Runs a Python function in a subprocess."
    input_model: ClassVar[type[BaseModel]] = _EmptyInputs

    class Config(_FunctionModel):
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
        timeout_s : float or None, default=None
            Execution timeout, or None to use the run's tool timeout.
        grace_s : float, default=10.0
            Shutdown grace period before forced termination.

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
        timeout_s: float | None = None
        grace_s: float = 10.0

    def __init__(
        self, function: "str | Callable[..., Any] | FunctionTool.Config", /, **fields: Any
    ) -> None:
        """Validate settings while retaining only the function's import path."""
        if isinstance(function, self.Config):
            data = config_or_fields(function, fields)
        else:
            path = import_path(function) if callable(function) else function
            data = {**fields, "function": path}
        super().__init__(_validate_config(self.Config, data, where="FunctionTool"))

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
        >>> model = FunctionTool("my_package.tools:solve").get_input_model()  # doctest: +SKIP
        """
        return function_input_model(_load_function(self.config.function))

    def preflight(self) -> None:
        """
        Check that the referenced function has a supported input signature.

        Examples
        --------
        >>> from swarm_arena import FunctionTool
        >>> FunctionTool("my_package.tools:solve").preflight()  # doctest: +SKIP
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
        fn = _load_function(self.config.function)
        return Command(
            argv=[
                "python",
                "-P",
                "-m",
                "swarm_arena.run_function",
                "--import-root",
                import_root(fn.__module__),
                self.config.function,
            ],
            timeout_s=self.config.timeout_s,
            grace_s=self.config.grace_s,
        )

    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        """
        Collect the function's result using the standard tool envelope rules.

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
        >>> output = tool.collect(request, workspace, run)  # doctest: +SKIP
        """
        return tool_output_from_result(workspace, run)
