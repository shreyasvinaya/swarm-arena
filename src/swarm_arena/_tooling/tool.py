"""Define tool connectors, typed manager actions, and shared timeout rules."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, TypeVar

from pydantic import BaseModel

from swarm_arena._model.base import JsonValue
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.ref import Configurable
from swarm_arena._model.tooling import Command, CompletedRun, Estimate, ToolOutput, ToolRequest

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
    >>> from swarm_arena.tool import Tool
    >>> sorted(Tool.__abstractmethods__)
    ['collect', 'command']
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
        >>> tool.preflight()  # doctest: +SKIP
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
        >>> from swarm_arena.tool import ToolRequest
        >>> tool.is_deterministic(ToolRequest())  # doctest: +SKIP
        True
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
        >>> tool.editable()  # doctest: +SKIP
        ()
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
        >>> from swarm_arena.tool import ToolRequest
        >>> tool.estimate(ToolRequest()) is None  # doctest: +SKIP
        True
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
        >>> tool.providers()  # doctest: +SKIP
        frozenset()
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
        >>> from swarm_arena.tool import RunStatus, ToolOutput
        >>> output = ToolOutput(status=RunStatus.OK, data={"a": 1})
        >>> len(tool.fingerprint(output))  # doctest: +SKIP
        64
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
        >>> tool.extra_actions()  # doctest: +SKIP
        ()
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
        >>> tool.get_input_model().model_validate({})  # doctest: +SKIP
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
        >>> tool.get_description()  # doctest: +SKIP
        'Runs Python code.'
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
        >>> command = tool.command(request, workspace)  # doctest: +SKIP
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
    return getattr(tool.config, "grace_s", 10.0)
