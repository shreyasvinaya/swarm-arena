"""Define tool connectors, typed manager actions, and shared timeout rules."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import AliasChoices, AliasPath, BaseModel

from swarm_arena._model.base import JsonValue
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.ref import Configurable, Ref
from swarm_arena._model.tooling import (
    DEFAULT_GRACE_S,
    Command,
    CompletedRun,
    Estimate,
    ToolOutput,
    ToolRequest,
)

InputT = TypeVar("InputT", bound=BaseModel)


@dataclass(frozen=True, kw_only=True)
class Action(Generic[InputT]):
    """
    Describe a typed asynchronous action offered to a manager.

    Parameters
    ----------
    name : str
        Action name exposed to the manager.
    description : str
        Explanation of the action's purpose.
    input_model : type[InputT]
        Model that validates the action's arguments.
    handler : Callable[[Any, InputT], Awaitable[JsonValue]]
        Handler receiving the agent's AgentContext and validated arguments.
        AgentContext becomes available in M1b.
    counts_as : {'board_read', 'none'}, default='none'
        Counter charged when the action runs.

    Examples
    --------
    >>> from typing import Any
    >>> from pydantic import BaseModel
    >>> from swarm_arena.tool import Action
    >>> class Inputs(BaseModel):
    ...     n: int
    >>> async def echo(context: Any, inputs: Inputs) -> int:
    ...     return inputs.n
    >>> action = Action(name="echo", description="Echo an integer.",
    ...                 input_model=Inputs, handler=echo)
    >>> action.counts_as
    'none'
    """

    name: str
    description: str
    input_model: type[InputT]
    handler: Callable[[Any, InputT], Awaitable[JsonValue]]
    counts_as: Literal["board_read", "none"] = "none"


class Tool(Configurable, ABC):
    """
    Describe a tool command and collect its completed or partial result.

    Construction validates configuration without performing I/O. Subclasses
    implement ``command`` and ``collect``; optional hooks provide metadata.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None, optional
        Settings validated by the subclass's nested ``Config`` model.
        None selects that model's defaults.

    Attributes
    ----------
    name : ClassVar[str]
        Tool name.
    description : ClassVar[str]
        Description shown to the manager.
    input_model : ClassVar[type[BaseModel]]
        Model validating tool inputs; prefer ``get_input_model`` on instances.

    Examples
    --------
    >>> from pathlib import Path
    >>> from pydantic import BaseModel
    >>> from swarm_arena.tool import (
    ...     Command, CompletedRun, RunStatus, Tool, ToolOutput, ToolRequest,
    ... )
    >>> sorted(Tool.__abstractmethods__)
    ['collect', 'command']

    A concrete tool can demonstrate the default hooks entirely in memory:

    >>> class Inputs(BaseModel):
    ...     n: int = 0
    >>> class ExampleTool(Tool):
    ...     name = "example"
    ...     description = "An in-memory example."
    ...     input_model = Inputs
    ...     def command(self, request: ToolRequest, workspace: Path) -> Command:
    ...         return Command(argv=["echo", "example"])
    ...     def collect(
    ...         self, request: ToolRequest, workspace: Path, run: CompletedRun,
    ...     ) -> ToolOutput:
    ...         return ToolOutput(status=run.status, data={})
    >>> tool = ExampleTool()
    >>> tool.preflight()
    >>> tool.is_deterministic(ToolRequest())
    True
    >>> tool.editable()
    ()
    >>> tool.estimate(ToolRequest()) is None
    True
    >>> tool.providers()
    frozenset()
    >>> output = ToolOutput(status=RunStatus.OK, data={"a": 1})
    >>> len(tool.fingerprint(output))
    64
    >>> tool.extra_actions()
    ()
    >>> tool.get_input_model().model_validate({}).model_dump()
    {'n': 0}
    >>> tool.get_description()
    'An in-memory example.'
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]

    def preflight(self) -> None:
        """
        Check readiness before running the tool.

        The default implementation performs no checks or I/O.

        Examples
        --------
        The :class:`Tool` example calls ``tool.preflight()`` without I/O.
        """

    def is_deterministic(self, request: ToolRequest) -> bool:
        """
        Report whether repeating a request should yield the same result.

        Parameters
        ----------
        request : ToolRequest
            Invocation whose determinism is being assessed.

        Returns
        -------
        bool
            True by default; override to permit stochastic replicates.

        Examples
        --------
        The :class:`Tool` example calls ``tool.is_deterministic(ToolRequest())``
        and returns True.
        """
        return True

    def editable(self) -> Sequence[str]:
        """
        Return workspace globs that the manager may edit.

        Returns
        -------
        Sequence[str]
            No editable files by default.

        Examples
        --------
        The :class:`Tool` example calls ``tool.editable()`` and returns ``()``.
        """
        return ()

    def estimate(self, request: ToolRequest) -> Estimate | None:
        """
        Estimate the cost of a tool request.

        Parameters
        ----------
        request : ToolRequest
            Invocation to estimate.

        Returns
        -------
        Estimate or None
            Estimated time, tokens or cost, or None when unknown.

        Examples
        --------
        The :class:`Tool` example calls ``tool.estimate(ToolRequest())`` and
        checks that the result is None.
        """
        return None

    def providers(self) -> frozenset[str]:
        """
        Return the language-model providers called by the tool.

        Returns
        -------
        frozenset[str]
            No providers by default.

        Examples
        --------
        The :class:`Tool` example calls ``tool.providers()`` and returns an
        empty frozenset.
        """
        return frozenset()

    def fingerprint(self, output: ToolOutput) -> str:
        """
        Hash canonical output data to identify distinct results.

        Parameters
        ----------
        output : ToolOutput
            Result whose structured data determines its identity.

        Returns
        -------
        str
            Lowercase SHA-256 digest of canonical JSON data.

        Examples
        --------
        The :class:`Tool` example calls ``tool.fingerprint(output)`` and checks
        the digest length.
        """
        return sha256_hex(canonical_json(output.data))

    def extra_actions(self) -> Sequence[Action[Any]]:
        """
        Return additional typed actions offered to the manager.

        Returns
        -------
        Sequence[Action[Any]]
            No extra actions by default.

        Examples
        --------
        The :class:`Tool` example calls ``tool.extra_actions()`` and returns ``()``.
        """
        return ()

    def get_input_model(self) -> type[BaseModel]:
        """
        Return the model used to validate this tool's inputs.

        Returns
        -------
        type[BaseModel]
            The subclass's input model, unless overridden per instance.

        Examples
        --------
        The :class:`Tool` example calls ``tool.get_input_model()`` and validates
        an empty mapping with the returned model.
        """
        return type(self).input_model

    def get_description(self) -> str:
        """
        Return the description shown to the manager.

        Returns
        -------
        str
            The subclass's description, unless overridden per instance.

        Examples
        --------
        The :class:`Tool` example calls ``tool.get_description()`` and returns
        the concrete tool's description.
        """
        return type(self).description

    @abstractmethod
    def command(self, request: ToolRequest, workspace: Path) -> Command:
        """
        Build the command to execute inside the staged workspace.

        Parameters
        ----------
        request : ToolRequest
            Validated tool invocation.
        workspace : Path
            Staged working directory on the executor's node.

        Returns
        -------
        Command
            Program, environment allowlist, and execution settings.

        Examples
        --------
        >>> from pathlib import Path
        >>> from swarm_arena import ScriptTool
        >>> from swarm_arena.tool import ToolRequest
        >>> ScriptTool(command=["true"]).command(ToolRequest(), Path(".")).argv
        ['true']
        """

    @abstractmethod
    def collect(self, request: ToolRequest, workspace: Path, run: CompletedRun) -> ToolOutput:
        """
        Collect output after execution, including timeout or cancellation.

        Parameters
        ----------
        request : ToolRequest
            Invocation originally passed to ``command``.
        workspace : Path
            Workspace holding result files and logs.
        run : CompletedRun
            Process outcome and timestamps.

        Returns
        -------
        ToolOutput
            Complete or partial output, usage, and declared artifacts.

        Examples
        --------
        >>> output = tool.collect(request, workspace, run)  # doctest: +SKIP
        """

    def _own_files(self) -> dict[str, str]:
        """Return the tool's own root files as paths mapped to artifact hashes."""
        return {}

    def _protected(self, root: AbstractSet[str]) -> frozenset[str]:
        """Return protected tool paths; the driver unions these with private paths."""
        return frozenset()


def effective_tool_ref(tool: Tool | Ref, overrides: Mapping[str, Any]) -> Ref:
    """
    Validate a shallow override merge while preserving raw tool settings.

    Parameters
    ----------
    tool : Tool or Ref
        Base tool configuration. Call only after raw planner authorization.
    overrides : Mapping[str, Any]
        Top-level replacements; human-only fields and validation aliases are
        forbidden even when their values are unchanged.

    Returns
    -------
    Ref
        Independent reference validated through Ref and the tool's Config.
        Raw scalar types, omitted defaults, and frozen manifests are retained.

    Raises
    ------
    ValueError
        If overrides change human-only fields or configuration is invalid.
    TypeError
        If the reference does not select a Tool subclass.

    Examples
    --------
    >>> ref = Ref(path="swarm_arena:ScriptTool", config={"command": ["run"]})
    >>> effective_tool_ref(ref, {"timeout_s": 12}).config
    {'command': ['run'], 'timeout_s': 12}
    """
    ref = tool if isinstance(tool, Ref) else Ref.of(tool)
    cls = ref.load()
    if not issubclass(cls, Tool):
        raise TypeError(f"{ref.path} is not a subclass of Tool")
    for name in getattr(cls, "_human_only", frozenset()):
        keys = {name}
        field = cls.Config.model_fields.get(name)
        if field is not None:
            alias = field.validation_alias
            if isinstance(alias, str):
                keys.add(alias)
            elif isinstance(alias, AliasPath):
                keys.add(alias.path[0])
            elif isinstance(alias, AliasChoices):
                keys.update(path[0] for path in alias.convert_to_aliases())
        if keys.intersection(overrides):
            raise ValueError(f"round_overrides: human-only field {name} cannot be overridden")
    effective = Ref(path=ref.path, name=ref.name, config={**ref.config, **overrides})
    effective._resolve_config(Tool)
    return effective.model_copy(deep=True)


def cap_timeout(configured: float | None, limit_s: float) -> float:
    """Apply the shared timeout default and cap without treating zero as missing."""
    return min(limit_s if configured is None else configured, limit_s)


def effective_tool_timeout(tool: Tool, *, round_timeout_s: float | None, limit_s: float) -> float:
    """Choose the round override, tool setting, or limit and apply the hard cap."""
    configured = (
        round_timeout_s if round_timeout_s is not None else getattr(tool.config, "timeout_s", None)
    )
    return cap_timeout(configured, limit_s)


def tool_grace(tool: Tool) -> float:
    """Return the tool's configured shutdown grace or the ten-second default."""
    return getattr(tool.config, "grace_s", DEFAULT_GRACE_S)
