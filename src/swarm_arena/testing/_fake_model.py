"""Scripted Agents SDK responses using the real call recorder."""

from collections import deque
from collections.abc import AsyncIterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from agents import AgentOutputSchemaBase, Handoff, Model, ModelResponse, ModelSettings, Tool
from agents.items import TResponseInputItem, TResponseOutputItem, TResponseStreamEvent
from agents.models.interface import ModelTracing
from agents.usage import Usage as SDKUsage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_prompt_param import ResponsePromptParam
from openai.types.responses.response_usage import InputTokensDetails

from swarm_arena._budget.llm_scope import current_llm_scope
from swarm_arena._errors import _core_error
from swarm_arena._journal.llm_record import pending_scope, record_call
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.data import finite_json
from swarm_arena._model.tooling import Usage
from swarm_arena.llm._catalog import call_cost, register_fake_model


@dataclass(frozen=True, kw_only=True)
class FakeTurn:
    """
    Describe one scripted assistant response and its authoritative usage.

    Parameters
    ----------
    text : str or None, default=None
        Assistant text, omitted when None.
    tool_calls : tuple[tuple[str, dict[str, Any]], ...], default=()
        Function names and finite JSON argument objects, in response order.
    usage : Usage, optional
        Token usage, defaulting to 100 input and 20 output tokens.

    Examples
    --------
    >>> from swarm_arena.testing import FakeTurn
    >>> turn = FakeTurn(text="done", tool_calls=(("run_tool", {"x": 1}),))
    >>> turn.text, turn.usage.total_tokens
    ('done', 120)
    """

    text: str | None = None
    tool_calls: tuple[tuple[str, dict[str, Any]], ...] = ()
    usage: Usage = Usage(input_tokens=100, output_tokens=20)  # noqa: RUF009 - Usage is frozen


class FakeModel(Model):
    """
    Replay isolated scripted conversations through the Agents SDK interface.

    Parameters
    ----------
    turns : Sequence[FakeTurn]
        Script copied at construction. Each scoped conversation or explicit SDK
        conversation ID replays its own copy; calls without either share one
        sequential conversation. A scope is identified by run, goal, team,
        agent, round, role and candidate.
    provider : str, default='fake'
        Provider identifier recorded with metered calls.

    Attributes
    ----------
    calls : list[dict[str, Any]]
        Received system, input, model_settings, tools, output_schema and tracing
        arguments, in invocation order across conversations.

    Notes
    -----
    Construct and register a fresh model for each independent test or run to
    reset call history and registrations. New scope run IDs start fresh queues
    even when an instance is reused. Streaming is unsupported.

    Examples
    --------
    >>> from swarm_arena.testing import FakeModel, FakeTurn
    >>> model = FakeModel([FakeTurn(text="hello")])
    >>> model.calls
    []
    """

    def __init__(self, turns: Sequence[FakeTurn], *, provider: str = "fake") -> None:
        self._turns = deepcopy(tuple(turns))
        self._conversations: dict[tuple[Any, ...], deque[FakeTurn]] = {}
        self._name = "unregistered"
        self._tool_count = 0
        self._response_count = 0
        self.provider = provider
        self.calls: list[dict[str, Any]] = []

    def register(self, name: str) -> None:
        """
        Register this instance for an offline model specification.

        Parameters
        ----------
        name : str
            Model name after the ``fake/`` prefix. Replaces an earlier
            registration of the same name without reusing its consumed queue.

        Examples
        --------
        >>> from swarm_arena.testing import FakeModel, FakeTurn
        >>> model = FakeModel([FakeTurn(text="ok")])
        >>> model.register("offline-example")
        """
        register_fake_model(name, self)
        self._name = name

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> ModelResponse:
        """
        Return the next turn and settle a metered reservation when supplied.

        Parameters
        ----------
        system_instructions : str or None
            System prompt retained in calls.
        input : str or list[TResponseInputItem]
            SDK input retained in calls.
        model_settings : ModelSettings
            Settings with optional recorder tracking metadata.
        tools : list[Tool]
            Available tools retained in calls.
        output_schema : AgentOutputSchemaBase or None
            Output contract retained in calls.
        handoffs : list[Handoff]
            SDK handoffs, accepted without interpretation.
        tracing : ModelTracing
            SDK tracing mode retained in calls.
        previous_response_id : str or None
            Accepted for SDK compatibility; no server-side response is stored.
        conversation_id : str or None
            Independent script key outside a component scope.
        prompt : ResponsePromptParam or None
            Hosted prompt accepted without interpretation.

        Returns
        -------
        ModelResponse
            Scripted messages, function calls and SDK token usage.

        Raises
        ------
        RuntimeError
            If this conversation has no turns left.
        ValueError
            If supplied text is not a string or function arguments are not
            finite JSON objects. The core error names only the failing field.

        Examples
        --------
        >>> import asyncio
        >>> from agents import ModelSettings
        >>> from agents.models.interface import ModelTracing
        >>> from swarm_arena.testing import FakeModel, FakeTurn
        >>> model = FakeModel([FakeTurn(text="ok")])
        >>> response = asyncio.run(model.get_response(
        ...     None, "hello", ModelSettings(), [], None, [], ModelTracing.DISABLED,
        ...     previous_response_id=None, conversation_id=None, prompt=None))
        >>> response.output[0].content[0].text
        'ok'
        """
        self.calls.append(
            {
                "system": system_instructions,
                "input": input,
                "model_settings": model_settings,
                "tools": tools,
                "output_schema": output_schema,
                "tracing": tracing,
            }
        )
        metadata = model_settings.metadata or {}
        pending = pending_scope(metadata) if "run_id" in metadata else None
        scope = pending or current_llm_scope()
        key = (
            (
                "scope",
                scope.run_id,
                scope.goal_id,
                scope.team,
                scope.agent,
                scope.round,
                scope.role,
                scope.candidate_id,
                conversation_id,
            )
            if scope is not None
            else ("sdk", conversation_id)
        )
        if key not in self._conversations:
            self._conversations[key] = deque(deepcopy(self._turns))
        turns = self._conversations[key]
        if not turns:
            raise RuntimeError("FakeModel has no turns left")
        turn = turns.popleft()
        if turn.text is not None and not isinstance(turn.text, str):
            raise _core_error(ValueError, "invalid text") from None
        prepared_calls: list[tuple[str, str]] = []
        for name, args in turn.tool_calls:
            if not isinstance(args, dict):
                raise _core_error(ValueError, "invalid arguments") from None
            try:
                arguments = canonical_json(finite_json(args)).decode("utf-8")
            except (TypeError, ValueError, OverflowError, RecursionError):
                raise _core_error(ValueError, "invalid arguments") from None
            prepared_calls.append((name, arguments))
        self._response_count += 1
        output: list[TResponseOutputItem] = []
        if turn.text is not None:
            output.append(
                ResponseOutputMessage(
                    id=f"fake-message-{self._response_count}",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(type="output_text", text=turn.text, annotations=[])
                    ],
                )
            )
        tool_ids: list[tuple[str, str]] = []
        for name, arguments in prepared_calls:
            self._tool_count += 1
            call_id = f"fake-{name}-{self._tool_count}"
            output.append(
                ResponseFunctionToolCall(
                    type="function_call",
                    name=name,
                    arguments=arguments,
                    call_id=call_id,
                )
            )
            tool_ids.append((call_id, name))
        usage = turn.usage
        response = ModelResponse(
            output=output,
            usage=SDKUsage(
                requests=1,
                input_tokens=usage.input_tokens
                + usage.cache_read_tokens
                + usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                input_tokens_details=InputTokensDetails(
                    cached_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                ),
            ),
            response_id=None,
        )
        if "run_id" in metadata:
            spec = pending.models.root[metadata["model_alias"]] if pending is not None else None
            request: dict[str, Any] = {"system": system_instructions, "input": input}
            await record_call(
                metadata=metadata,
                model=spec.model if spec is not None else f"fake/{self._name}",
                provider=self.provider,
                usage=usage,
                cost_usd=call_cost(spec, usage, litellm_cost=None) if spec is not None else 0,
                cache_hit=False,
                latency_ms=0,
                status="ok",
                error=None,
                request=request,
                response={"output": [item.model_dump(mode="json") for item in output]},
                litellm_call_id=None,
                tool_call_ids=tool_ids,
                settle=True,
            )
        return response

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: ResponsePromptParam | None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """
        Reject streaming calls with the SDK-compatible method signature.

        Parameters
        ----------
        system_instructions : str or None
            System prompt.
        input : str or list[TResponseInputItem]
            SDK input.
        model_settings : ModelSettings
            Call settings.
        tools : list[Tool]
            Available tools.
        output_schema : AgentOutputSchemaBase or None
            Output contract.
        handoffs : list[Handoff]
            Available handoffs.
        tracing : ModelTracing
            SDK tracing mode.
        previous_response_id : str or None
            Previous response identifier.
        conversation_id : str or None
            Conversation identifier.
        prompt : ResponsePromptParam or None
            Hosted prompt.

        Returns
        -------
        AsyncIterator[TResponseStreamEvent]
            Never returned because streaming is unsupported.

        Raises
        ------
        NotImplementedError
            Always; FakeModel does not stream.

        Examples
        --------
        >>> from agents import ModelSettings
        >>> from agents.models.interface import ModelTracing
        >>> from swarm_arena.testing import FakeModel
        >>> FakeModel([]).stream_response(
        ...     None, "hello", ModelSettings(), [], None, [], ModelTracing.DISABLED,
        ...     previous_response_id=None, conversation_id=None, prompt=None)
        Traceback (most recent call last):
        ...
        NotImplementedError: FakeModel does not stream
        """
        raise NotImplementedError("FakeModel does not stream")
