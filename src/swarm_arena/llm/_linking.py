"""Link SDK function invocations to their originating journal call."""

from dataclasses import dataclass

from agents import ModelResponse
from openai.types.responses import ResponseFunctionToolCall

from swarm_arena._agent.context import AgentContext
from swarm_arena._model.context_strings import _context_string


def tool_calls_of(response: ModelResponse) -> list[tuple[str, str]]:
    """
    Extract function call identifiers and names in response order.

    Parameters
    ----------
    response : agents.ModelResponse
        SDK response; custom model responses must be prepared off-loop.

    Returns
    -------
    list[tuple[str, str]]
        Plain call identifiers and function names.

    Examples
    --------
    >>> from agents.usage import Usage
    >>> tool_calls_of(ModelResponse(output=[], usage=Usage(), response_id=None))
    []
    """
    return [
        (_context_string(item.call_id, "tool_call_id"), _context_string(item.name, "name"))
        for item in response.output
        if isinstance(item, ResponseFunctionToolCall)
    ]


class CallLinks:
    """
    Retain tool-call to LLM-call links for one component invocation.

    Examples
    --------
    >>> CallLinks().get("unknown") is None
    True
    """

    def __init__(self) -> None:
        self._links: dict[str, str] = {}

    def add(self, call_id: str, response: ModelResponse) -> None:
        """
        Link each function call in a detached SDK response.

        Parameters
        ----------
        call_id : str
            Preallocated journal LLM call identifier.
        response : agents.ModelResponse
            Response whose function calls requested the actions.

        Examples
        --------
        >>> from agents.usage import Usage
        >>> links = CallLinks()
        >>> links.add("llm-1", ModelResponse(output=[], usage=Usage(), response_id=None))
        >>> links.get("unknown") is None
        True
        """
        call_id = _context_string(call_id, "call_id")
        self._links.update((tool_id, call_id) for tool_id, _ in tool_calls_of(response))

    def get(self, tool_call_id: str) -> str | None:
        """
        Look up the LLM call that requested an action.

        Parameters
        ----------
        tool_call_id : str
            SDK function call identifier.

        Returns
        -------
        str or None
            Journal call identifier, or None for an unknown action.

        Examples
        --------
        >>> CallLinks().get("unknown") is None
        True
        """
        return self._links.get(_context_string(tool_call_id, "tool_call_id"))


@dataclass(kw_only=True)
class ManagerDeps:
    """
    Carry manager state for Runner.run without sending it to the model.

    Parameters
    ----------
    ctx : AgentContext
        Authorized core action context.
    links : CallLinks
        Invocation-local tool-call linkage.
    summary : str or None, default=None
        Manager's optional completion summary.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> ManagerDeps(ctx=Mock(spec=AgentContext), links=CallLinks()).summary is None
    True
    """

    ctx: AgentContext
    links: CallLinks
    summary: str | None = None
