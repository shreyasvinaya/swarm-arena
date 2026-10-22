"""Meter model requests and transport vetted controls through the SDK."""

import json
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Any, TypeVar

import anyio
import litellm
from agents import (
    Agent,
    AgentOutputSchemaBase,
    FunctionTool,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    RunContextWrapper,
    RunHooks,
    Tool,
)
from agents.exceptions import AgentsException
from agents.items import TResponseInputItem, TResponseStreamEvent
from agents.models.chatcmpl_converter import Converter
from agents.models.interface import ModelTracing
from openai.types.responses.response_prompt_param import ResponsePromptParam
from pydantic import TypeAdapter

from swarm_arena._agent.manager import LimitReached
from swarm_arena._budget.amount import Amount
from swarm_arena._budget.llm_scope import LLMScope, current_llm_scope
from swarm_arena._errors import (
    Fenced,
    _core_error,
    _is_core_control,
    _register_carrier,
    _register_carrier_type,
)
from swarm_arena._errors import _unwrap_control as _unwrap_control
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.llm_record import abandon_call, open_call
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.catalog import ModelSpec
from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.data import finite_json
from swarm_arena._outcomes import _classify_outcome, _Outcome
from swarm_arena.llm._cache import cache_namespace, cache_switches
from swarm_arena.llm._catalog import call_estimate
from swarm_arena.llm._linking import CallLinks, ManagerDeps
from swarm_arena.llm._settings import merge_model_settings
from swarm_arena.llm._settings import model_settings as per_call_settings


class BudgetExceeded(AgentsException, LimitReached):
    """
    Report a refused LLM budget reservation through the Agents SDK.

    Parameters
    ----------
    limit : str
        Limit identifier; budget refusals use ``"budget"``.
    message : str
        Description of the refusal.

    Notes
    -----
    Instances constructed by callers do not carry core control authority.
    The metered adapter privately retains the original vetted core refusal.

    Examples
    --------
    >>> from swarm_arena.llm import BudgetExceeded
    >>> from swarm_arena.manager import LimitReached
    >>> error = BudgetExceeded("budget", "LLM budget used up")
    >>> isinstance(error, LimitReached), error.limit, str(error)
    (True, 'budget', 'LLM budget used up')
    """

    def __init__(self, limit: str, message: str) -> None:
        LimitReached.__init__(self, limit, message)
        self.message = message
        self.run_data = None


class _SDKControl(AgentsException):
    """Cross an SDK boundary without granting its mutable fields authority."""


_register_carrier_type(BudgetExceeded, "BudgetExceeded")
_register_carrier_type(_SDKControl, "RuntimeError")


def _sdk_control(error: BaseException) -> AgentsException | None:
    """Carry only registered controls; use the shared precedence for groups.

    Action adapters must first classify with their captured _Outcome (actual
    caller cancellation wins), then raise this result from None when present.
    Set SDK function tools' failure_error_function=None to prevent conversion
    of a carrier into a tool-error response. Ordinary failures return None.
    """
    try:
        core = _classify_outcome(error)
    except Fenced as fenced:
        core = fenced
    if not _is_core_control(core):
        return None
    carrier = (
        BudgetExceeded("budget", str(core))
        if type(core) is LimitReached and core.limit == "budget"
        else _SDKControl(str(core))
    )
    _register_carrier(carrier, core)
    return carrier


_T = TypeVar("_T")


async def _prepare(operation: Callable[[], _T]) -> _T:
    """Prepare untrusted SDK data under the caller's enclosing supervision."""
    outcome = _Outcome()
    try:
        return await run_sync_daemon(operation)
    except BaseException as error:
        outcome.contain(error, ValueError, "LLM preparation")


@lru_cache(maxsize=1)
def _response_adapter() -> TypeAdapter[ModelResponse]:
    return TypeAdapter(ModelResponse)


def _detach_response(response: ModelResponse) -> ModelResponse:
    """Round-trip through SDK base types in the abandonable preparation worker."""
    adapter = _response_adapter()
    return adapter.validate_json(adapter.dump_json(response, warnings="error"))


@dataclass(frozen=True)
class _RequestSchema(AgentOutputSchemaBase):
    """Pure request schema; Runner retains its original output validator."""

    schema: dict[str, Any]
    schema_name: str
    strict: bool

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return self.schema_name

    def json_schema(self) -> dict[str, Any]:
        return self.schema

    def is_strict_json_schema(self) -> bool:
        return self.strict

    def validate_json(self, json_str: str) -> Any:
        raise NotImplementedError("request schema does not validate model output")


def _json_copy(value: Any) -> Any:
    """Return plain finite JSON from an untrusted SDK request field."""
    return json.loads(canonical_json(finite_json(value)))


@dataclass(frozen=True)
class _Request:
    system: str | None
    items: str | list[TResponseInputItem]
    tools: list[Tool]
    output_schema: AgentOutputSchemaBase | None
    handoffs: list[Handoff]
    prompt: ResponsePromptParam | None
    chars: int


def _prepare_request(
    system: str | None,
    items: str | Sequence[TResponseInputItem],
    tools: list[Tool],
    output_schema: AgentOutputSchemaBase | None,
    handoffs: list[Handoff],
    prompt: ResponsePromptParam | None = None,
) -> _Request:
    """Detach the exact data counted for admission before the SDK reuses it."""
    system = _context_string(system, "system_prompt") if system is not None else None
    items = _json_copy(items if isinstance(items, str) else list(items))
    prompt = _json_copy(prompt)
    converted = [_json_copy(Converter.tool_to_openai(tool))["function"] for tool in tools]
    plain_tools: list[Tool] = []
    for tool, schema in zip(tools, converted, strict=True):
        assert isinstance(tool, FunctionTool)
        plain_tools.append(
            FunctionTool(
                name=schema["name"],
                description=schema["description"],
                params_json_schema=schema["parameters"],
                strict_json_schema=schema["strict"],
                on_invoke_tool=tool.on_invoke_tool,
            )
        )
    converted_handoffs = [
        _json_copy(Converter.convert_handoff_tool(handoff))["function"] for handoff in handoffs
    ]
    plain_handoffs = [
        Handoff(
            tool_name=schema["name"],
            tool_description=schema["description"],
            input_json_schema=schema["parameters"],
            strict_json_schema=schema["strict"],
            on_invoke_handoff=handoff.on_invoke_handoff,
            agent_name=_context_string(handoff.agent_name, "agent_name"),
        )
        for handoff, schema in zip(handoffs, converted_handoffs, strict=True)
    ]
    plain_schema = None
    if output_schema is not None and not output_schema.is_plain_text():
        plain_schema = _RequestSchema(
            _json_copy(output_schema.json_schema()),
            _context_string(output_schema.name(), "output_schema.name"),
            bool(output_schema.is_strict_json_schema()),
        )
    request = {
        "system": system,
        "input": items,
        "tools": [Converter.tool_to_openai(tool) for tool in plain_tools],
        "handoffs": [Converter.convert_handoff_tool(handoff) for handoff in plain_handoffs],
        "output_schema": plain_schema.schema if plain_schema is not None else None,
        "prompt": prompt,
    }
    chars = len(canonical_json(finite_json(request)).decode("utf-8"))
    return _Request(system, items, plain_tools, plain_schema, plain_handoffs, prompt, chars)


def _prepare_settings(settings: ModelSettings) -> ModelSettings:
    """Snapshot SDK settings and detach provider mappings before loop use."""
    if litellm.model_fallbacks is not None:
        raise _core_error(ValueError, "global model fallbacks are unsupported for metered calls")
    values = {field.name: getattr(settings, field.name) for field in fields(ModelSettings)}
    for field in ("metadata", "extra_args", "extra_body", "extra_query", "extra_headers"):
        if values.get(field) is not None:
            values[field] = _json_copy(values[field])
    metadata = values.get("metadata")
    if metadata is not None:
        values["metadata"] = {
            _context_string(key, "metadata key"): _context_string(value, "metadata value")
            for key, value in metadata.items()
        }
    return ModelSettings(**values)


# LiteLLM 1.107: main/utils retry wrappers, router request controls and
# RUNTIME_UPDATABLE_ROUTER_SETTINGS. Keep the corresponding upstream facts
# current when upgrading. SDK ModelSettings.retry belongs to Runner, whose
# retries re-enter get_response and receive separate reservations.
_PROVIDER_ATTEMPT_CONTROLS = frozenset(
    {
        "num_retries",
        "max_retries",
        "retry_policy",
        "model_group_retry_policy",
        "retry_strategy",
        "retry_after",
        "fallbacks",
        "context_window_fallback_dict",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "default_fallbacks",
        "max_fallbacks",
        "fallback_on_unsupported_params",
        "disable_fallbacks",
    }
)


def _without_retries(settings: ModelSettings) -> ModelSettings:
    """Strip hidden-attempt controls, then set authoritative zero retry counts."""

    def strip(values: dict[str, Any]) -> dict[str, Any]:
        return {
            key: strip(value)
            if key in {"extra_args", "extra_body"} and isinstance(value, dict)
            else value
            for key, value in values.items()
            if key not in _PROVIDER_ATTEMPT_CONTROLS
        }

    values = strip({field.name: getattr(settings, field.name) for field in fields(ModelSettings)})
    # Omitting numeric controls would allow LiteLLM's process defaults to win.
    values["extra_args"] = {**(values["extra_args"] or {}), "num_retries": 0, "max_retries": 0}
    return ModelSettings(**values)


class MeteredModel(Model):
    """
    Reserve each non-streaming model call against the current LLM scope.

    Parameters
    ----------
    inner : agents.Model
        SDK model or registered offline fake.
    alias : str
        Approved catalog alias.
    spec : ModelSpec
        Catalog specification and output ceiling.
    namespace : str
        Goal namespace, overridden by an explicit cache namespace.
    links : CallLinks or None, optional
        Invocation-local tool linkage.
    no_cache_read : bool, default=False
        Suppress response-cache reads for this instance.
    temperature : float or None, optional
        Per-call sampling setting.
    max_tokens : int or None, optional
        Output cap, never larger than the catalog cap.

    admission_check : Callable[[], None] or None, optional
        Component gate checked before funding and again before call registration.
        Already registered calls complete normally.

    Attributes
    ----------
    call_ids : list[str]
        Preallocated identifiers for calls admitted through this instance.

    Examples
    --------
    >>> from swarm_arena.testing import FakeModel
    >>> model = MeteredModel(FakeModel([]), alias="m",
    ...     spec=ModelSpec(model="fake/m", roles=["manager"]), namespace="n")
    >>> model.call_ids
    []
    """

    def __init__(
        self,
        inner: Model,
        *,
        alias: str,
        spec: ModelSpec,
        namespace: str,
        links: CallLinks | None = None,
        no_cache_read: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        admission_check: Callable[[], None] | None = None,
    ) -> None:
        self.inner = inner
        self.alias = _context_string(alias, "alias")
        self.spec = spec
        self.namespace = _context_string(namespace, "namespace")
        self.links = links
        self.no_cache_read = no_cache_read
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.admission_check = admission_check
        self.call_ids: list[str] = []

    def _cap(self, caller: ModelSettings) -> int:
        cap = self.max_tokens or caller.max_tokens or self.spec.max_tokens
        if type(cap) is not int or cap <= 0:
            raise _core_error(ValueError, "invalid max_tokens")
        return min(cap, self.spec.max_tokens)

    def precheck_estimate(
        self, system_prompt: str | None, input_items: Sequence[TResponseInputItem]
    ) -> Amount:
        """
        Estimate the advisory hook input using this model's output cap.

        Parameters
        ----------
        system_prompt : str or None
            SDK system prompt.
        input_items : Sequence[TResponseInputItem]
            SDK input items. Hooks run serialization in a preparation worker.

        Returns
        -------
        Amount
            Advisory input/output estimate; reservation counts tools and output
            schemas too, and remains authoritative.

        Examples
        --------
        >>> from swarm_arena.testing import FakeModel
        >>> model = MeteredModel(FakeModel([]), alias="m",
        ...     spec=ModelSpec(model="fake/m", roles=["manager"]), namespace="n")
        >>> model.precheck_estimate(None, []).tokens >= 4096
        True
        """
        return call_estimate(
            self.spec,
            prompt_chars=_prepare_request(system_prompt, input_items, [], None, []).chars,
            max_tokens=self._cap(ModelSettings()),
        )

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
        Reserve, record tracking metadata and delegate one SDK model attempt.

        Parameters
        ----------
        system_instructions : str or None
            System prompt.
        input : str or list[TResponseInputItem]
            SDK input.
        model_settings : agents.ModelSettings
            Caller settings, resolved before the hook by the SDK.
        tools : list[Tool]
            Available tool schemas.
        output_schema : AgentOutputSchemaBase or None
            Structured output contract.
        handoffs : list[Handoff]
            Available handoff schemas.
        tracing : ModelTracing
            SDK tracing mode.
        previous_response_id, conversation_id : str or None
            SDK conversation identifiers.
        prompt : ResponsePromptParam or None
            Hosted prompt configuration.

        Returns
        -------
        agents.ModelResponse
            Detached response with invocation-local call links recorded.

        Raises
        ------
        RuntimeError
            If no LLM scope is active.
        BudgetExceeded
            If the current tranche refuses the reservation for budget.

        Examples
        --------
        Scope owners call this through Runner.run, then drain their scope
        before closing its tranche. See ``precheck_estimate`` for an in-memory
        funding example.
        """
        scope = current_llm_scope()
        if scope is None:
            raise RuntimeError("LLM call outside an LLM scope")

        def prepare() -> tuple[_Request, ModelSettings, ModelSpec, int, Amount]:
            request = _prepare_request(
                system_instructions, input, tools, output_schema, handoffs, prompt
            )
            caller = _prepare_settings(model_settings)
            spec = ModelSpec.model_validate_json(self.spec.model_dump_json())
            cap = self._cap(caller)
            estimate = call_estimate(spec, prompt_chars=request.chars, max_tokens=cap)
            return request, caller, spec, cap, estimate

        request, caller, spec, cap, estimate = await _prepare(prepare)
        previous_response_id = (
            _context_string(previous_response_id, "previous_response_id")
            if previous_response_id is not None
            else None
        )
        conversation_id = (
            _context_string(conversation_id, "conversation_id")
            if conversation_id is not None
            else None
        )
        if self.admission_check is not None:
            self.admission_check()
        outcome = _Outcome()
        try:
            rid = await scope.tranche.reserve_call(estimate)
        except BaseException as error:
            classified = outcome.classify(_unwrap_control(error))
            carrier = _sdk_control(classified)
            raise carrier if carrier is not None else classified from None
        try:
            if self.admission_check is not None:
                self.admission_check()
            call_id = open_call(
                scope,
                reservation_id=rid,
                estimate=estimate,
                model_alias=self.alias,
                model=spec.model,
                provider=spec.provider,
            )
        except BaseException:
            # Registration refused before provider dispatch: no usage was possible.
            # Keep the refusal if durable accounting also fails; owner cleanup
            # retains the reservation and genuine cancellation still propagates.
            with suppress(Exception):
                await scope.tranche.settle_call(rid, Amount())
            raise
        self.call_ids.append(call_id)

        def settings() -> ModelSettings:
            read, write = (
                (False, False)
                if scope.role == "tool"
                else cache_switches(
                    scope.cache,
                    scope.role,
                    force_no_read=self.no_cache_read or scope.force_no_cache_read,
                    streamed=False,
                    thinking=caller.reasoning is not None,
                )
            )
            ours = per_call_settings(
                spec,
                alias=self.alias,
                scope=scope,
                reservation_id=rid,
                call_id=call_id,
                namespace=cache_namespace(scope.cache, self.namespace),
                read=read,
                write=write,
                max_tokens=cap,
                temperature=self.temperature,
            )
            return _without_retries(merge_model_settings(caller, ours))

        request_outcome = _Outcome()
        try:
            sent = await _prepare(settings)
            response = await self.inner.get_response(
                request.system,
                request.items,
                sent,
                request.tools,
                request.output_schema,
                request.handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=request.prompt,
            )
            response = await _prepare(lambda: _detach_response(response))
        except BaseException as error:
            try:
                classified = request_outcome.classify(_unwrap_control(error))
            except anyio.get_cancelled_exc_class():
                abandon_call(scope, rid)
                raise
            raise classified from None
        if self.links is not None:
            self.links.add(call_id, response)
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
        Reject streaming with the SDK-compatible signature.

        Parameters
        ----------
        system_instructions : str or None
            System prompt.
        input : str or list[TResponseInputItem]
            SDK input.
        model_settings : agents.ModelSettings
            Caller settings.
        tools : list[Tool]
            Available tools.
        output_schema : AgentOutputSchemaBase or None
            Output contract.
        handoffs : list[Handoff]
            Available handoffs.
        tracing : ModelTracing
            SDK tracing mode.
        previous_response_id, conversation_id : str or None
            SDK conversation identifiers.
        prompt : ResponsePromptParam or None
            Hosted prompt.

        Returns
        -------
        AsyncIterator[TResponseStreamEvent]
            Never returned; streaming is unsupported.

        Raises
        ------
        NotImplementedError
            Always; MeteredModel does not stream.

        Examples
        --------
        >>> from swarm_arena.testing import FakeModel
        >>> model = MeteredModel(FakeModel([]), alias="m",
        ...     spec=ModelSpec(model="fake/m", roles=["manager"]), namespace="n")
        >>> model.stream_response(None, [], ModelSettings(), [], None, [],
        ...     ModelTracing.DISABLED, previous_response_id=None,
        ...     conversation_id=None, prompt=None)
        Traceback (most recent call last):
            ...
        NotImplementedError: MeteredModel does not stream
        """
        raise NotImplementedError("MeteredModel does not stream")


class BudgetHooks(RunHooks[ManagerDeps]):
    """
    Check committed tranche credit before the SDK enters a metered call.

    Examples
    --------
    >>> callable(BudgetHooks().on_llm_start)
    True
    """

    async def on_llm_start(
        self,
        context: RunContextWrapper[ManagerDeps],
        agent: Agent[ManagerDeps],
        system_prompt: str | None,
        input_items: list[TResponseInputItem],
    ) -> None:
        """
        Apply an advisory precheck against the current scope's tranche.

        Parameters
        ----------
        context : RunContextWrapper[ManagerDeps]
            SDK context; its AgentContext budget is never read here.
        agent : Agent[ManagerDeps]
            Agent whose MeteredModel supplies the estimate.
        system_prompt : str or None
            System prompt.
        input_items : list[TResponseInputItem]
            SDK input items.

        Raises
        ------
        BudgetExceeded
            If neither committed credit nor remaining funding covers the call.
        RuntimeError
            If a metered model has no current LLM scope.

        Examples
        --------
        Unmetered models require no LLM scope for this hook.

        >>> from unittest.mock import Mock
        >>> from swarm_arena.testing import FakeModel
        >>> async def example() -> None:
        ...     deps = ManagerDeps(ctx=Mock(), links=CallLinks())
        ...     agent = Agent[ManagerDeps](name="example", model=FakeModel([]))
        ...     await BudgetHooks().on_llm_start(RunContextWrapper(deps), agent, None, [])
        >>> anyio.run(example)
        """
        model = agent.model
        if not isinstance(model, MeteredModel):
            return
        scope: LLMScope | None = current_llm_scope()
        if scope is None:
            raise RuntimeError("LLM call outside an LLM scope")
        estimate = await _prepare(lambda: model.precheck_estimate(system_prompt, input_items))
        if not scope.tranche.can_cover(estimate):
            # Ask the existing core overflow seam before creating a normal
            # budget refusal: a terminal overflow must retain its original reason.
            try:
                scope.tranche._ledger._check_usage_overflow()
            except LimitReached as error:
                carrier = _sdk_control(error)
                assert carrier is not None
                raise carrier from None
            carrier = _sdk_control(_core_error(LimitReached, "budget", "LLM budget used up"))
            assert carrier is not None
            raise carrier from None
