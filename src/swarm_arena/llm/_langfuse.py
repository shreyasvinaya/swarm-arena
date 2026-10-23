"""Own optional process-wide export while keeping consent local to each scope."""

import asyncio
import atexit
import builtins
import logging
import threading
import time
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any

from swarm_arena._budget.llm_scope import LLMScope, current_llm_scope
from swarm_arena._errors import ConfigError, _core_error, _exception_type_name
from swarm_arena._model.settings import TracingConfig
from swarm_arena._outcomes import _Outcome

_EXIT_TIMEOUT_S = 5.0
_PROJECT_TIMEOUT_S = 5
_lock = threading.Lock()
_log = logging.getLogger(__name__)


@dataclass
class _Resources:
    client: Any
    provider: Any
    instrumentor: Any
    requested: threading.Event = field(default_factory=threading.Event, init=False)
    worker: threading.Thread = field(init=False)
    project_worker: threading.Thread | None = field(default=None, init=False)
    project_url: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Python 3.12+ cannot start threads in atexit. Start the idle daemon now.
        self.worker = threading.Thread(
            target=self._wait_and_close, name="swarm-arena-tracing-close", daemon=True
        )
        self.worker.start()

    def _wait_and_close(self) -> None:
        self.requested.wait()
        _shutdown_resources(self)


_resources: _Resources | None = None
_closing: _Resources | None = None


def _resolve_project(resources: _Resources) -> None:
    """Resolve optional link metadata once, off the model-call path."""
    try:
        response = resources.client.api.projects.get(
            request_options={"max_retries": 0, "timeout_in_seconds": _PROJECT_TIMEOUT_S}
        )
        project_id = response.data[0].id if response.data else None
        if type(project_id) is not str or not project_id:
            raise ValueError("project identifier unavailable")
        resources.project_url = f"{resources.client._base_url}/project/{project_id}/traces/"
    except BaseException as error:
        # A missing prefix caches failure: neither setup nor calls retry it.
        _log.warning("%s", _exception_type_name(error))


def _shutdown_resources(resources: _Resources) -> None:
    """Close only owned resources, without recorder or journal callbacks."""
    from agents.tracing import setup

    def clear_processors() -> None:
        provider = setup.GLOBAL_TRACE_PROVIDER
        if provider is not None:
            provider.set_processors([])

    def forget() -> None:
        owned = resources.client._resources
        if owned is not None:
            # The pinned SDK retains shutdown resources in its key cache.
            with owned._lock:
                if owned._instances.get(owned.public_key) is owned:
                    del owned._instances[owned.public_key]

    def uninstrument() -> None:
        if resources.instrumentor.is_instrumented_by_opentelemetry:
            resources.instrumentor.uninstrument()

    def close_prompt_cache() -> None:
        owned = resources.client._resources
        if owned is not None:
            owned.prompt_cache._task_manager.shutdown()

    operations = (
        forget,
        clear_processors,
        uninstrument,
        close_prompt_cache,
        lambda: resources.client.flush(),
        lambda: resources.client.shutdown(),
        lambda: resources.provider.shutdown(),
    )
    for operation in operations:
        try:
            operation()
        except BaseException as error:
            _log.warning("%s", _exception_type_name(error))


def _close(resources: _Resources, timeout_s: float) -> None:
    """Signal the already-running daemon and bound the wait for export cleanup."""
    resources.requested.set()
    resources.worker.join(timeout_s)
    if resources.worker.is_alive():
        _log.warning("TimeoutError")


def _reset_langfuse(timeout_s: float) -> None:
    """Release the process owner with a bounded wait; used only at exit/tests."""
    global _resources, _closing
    deadline = time.monotonic() + timeout_s
    if not _lock.acquire(timeout=max(0, timeout_s)):
        _log.warning("TimeoutError")
        return
    try:
        resources = _resources or _closing
        _resources = None
        _closing = resources
        atexit.unregister(_close_at_exit)
        if resources is not None:
            _close(resources, max(0, deadline - time.monotonic()))
            if not resources.worker.is_alive():
                _closing = None
    finally:
        _lock.release()


def _close_at_exit() -> None:
    _reset_langfuse(_EXIT_TIMEOUT_S)


def _trace_error_kind(value: Any) -> str:
    """Accept known exception identifiers, never infer a type from error prose."""
    if type(value) is str:
        name = value.rsplit(".", 1)[-1]
        kind = vars(builtins).get(name)
        if isinstance(kind, type) and issubclass(kind, BaseException):
            return name
        if name == "CancelledError":
            return name
    return "Exception"


def _trace_error_key(name: str) -> bool:
    lowered = name.lower()
    return "error" in lowered or "exception" in lowered or "status_message" in lowered


def _trace_attributes(attributes: Any, *, is_error: bool = False) -> dict[str, Any]:
    values = dict(attributes or {})
    clean = {key: value for key, value in values.items() if not _trace_error_key(key)}
    if is_error or any(
        key.lower().startswith("exception.") or key.lower() == "error.type" for key in values
    ):
        clean["exception.type"] = _trace_error_kind(
            values.get("exception.type", values.get("error.type"))
        )
    return clean


def _error_boundary() -> Any:
    """Place error normalization ahead of every owned OTel processor."""
    from opentelemetry.sdk.trace import Event, ReadableSpan, SynchronousMultiSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    class DetachedSpan(ReadableSpan):
        # ReadableSpan derives these counters from its bounded containers, which
        # are replaced by the sanitized attribute/event copies below.
        def __init__(self, source: Any, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._dropped_counts = (
                source.dropped_attributes,
                source.dropped_events,
                source.dropped_links,
            )

        @property
        def dropped_attributes(self) -> int:
            return self._dropped_counts[0]

        @property
        def dropped_events(self) -> int:
            return self._dropped_counts[1]

        @property
        def dropped_links(self) -> int:
            return self._dropped_counts[2]

    class ErrorBoundary(SynchronousMultiSpanProcessor):
        def on_start(self, span: Any, parent_context: Any = None) -> None:
            attributes = _trace_attributes(
                span.attributes, is_error=span.status.status_code is StatusCode.ERROR
            )
            # OTel has no public attribute-removal API. This is the owned SDK
            # span's still-mutable attribute store, before downstream on_start.
            for key in tuple(span.attributes or {}):
                if _trace_error_key(key):
                    del span._attributes[key]
            span.set_attributes(attributes)
            super().on_start(span, parent_context=parent_context)

        def _on_ending(self, span: Any) -> None:
            # Owned Langfuse processing uses on_start/on_end. Do not expose the
            # unsanitized mutable ending span to an intermediate callback.
            pass

        def on_end(self, span: Any) -> None:
            events = []
            for event in span.events:
                if _trace_error_key(event.name):
                    attributes = {
                        "exception.type": _trace_error_kind(
                            (event.attributes or {}).get("exception.type")
                        )
                    }
                    events.append(Event("exception", attributes, event.timestamp))
                else:
                    events.append(
                        Event(event.name, _trace_attributes(event.attributes), event.timestamp)
                    )
            status = span.status
            if status.status_code is StatusCode.ERROR or status.description:
                status = Status(status.status_code, _trace_error_kind(status.description))
            clean = DetachedSpan(
                span,
                name=span.name,
                context=span.context,
                parent=span.parent,
                resource=span.resource,
                attributes=_trace_attributes(
                    span.attributes, is_error=status.status_code is StatusCode.ERROR
                ),
                events=events,
                links=span.links,
                kind=span.kind,
                status=status,
                start_time=span.start_time,
                end_time=span.end_time,
                instrumentation_scope=span.instrumentation_scope,
            )
            super().on_end(clean)

    return ErrorBoundary()


def setup_langfuse(tracing: TracingConfig) -> None:
    """Initialize the exclusive process exporter once, only after explicit opt-in.

    Parameters
    ----------
    tracing : TracingConfig
        Run configuration. Consent for individual calls remains on LLMScope.

    Raises
    ------
    ConfigError
        If the optional Langfuse dependencies are absent; includes the install hint.
    RuntimeError
        If tracing initialization fails; only the exception type is reported.

    Examples
    --------
    Disabled setup imports no optional tracing dependencies:

    >>> setup_langfuse(TracingConfig())
    >>> current_trace_url() is None
    True
    """
    global _resources, _closing
    with _lock:
        # Avoid the SDK's lazy default OpenAI exporter even on the first off run.
        from agents.tracing import setup
        from agents.tracing.provider import DefaultTraceProvider

        if setup.GLOBAL_TRACE_PROVIDER is None:
            setup.set_trace_provider(DefaultTraceProvider())
        if not tracing.langfuse:
            return
        if _closing is not None:
            if _closing.worker.is_alive():
                raise _core_error(RuntimeError, "Langfuse cleanup pending")
            _closing = None
        if _resources is not None:
            return
        try:
            from langfuse import Langfuse
            from openinference.instrumentation import TraceConfig
            from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
        except ImportError:
            raise _core_error(
                ConfigError,
                "Langfuse tracing requires swarm-arena[langfuse]: "
                'uv add "swarm-arena[langfuse] @ git+https://github.com/shreyasvinaya/swarm-arena"',
            ) from None

        class ConsentSampler(Sampler):
            def should_sample(
                self,
                parent_context: Any,
                trace_id: int,
                name: str,
                kind: Any = None,
                attributes: Any = None,
                links: Any = None,
                trace_state: Any = None,
            ) -> SamplingResult:
                scope = current_llm_scope()
                enabled = scope is not None and scope.tracing.langfuse
                return SamplingResult(
                    Decision.RECORD_AND_SAMPLE if enabled else Decision.DROP,
                    attributes=attributes,
                    trace_state=trace_state,
                )

            def get_description(self) -> str:
                return "swarm-arena scope consent"

        instrumentor = OpenAIAgentsInstrumentor()
        if instrumentor.is_instrumented_by_opentelemetry:
            raise _core_error(RuntimeError, "Langfuse instrumentor already owned")
        provider = TracerProvider(
            sampler=ConsentSampler(),
            shutdown_on_exit=False,
            active_span_processor=_error_boundary(),
        )
        client: Any = None
        resources: _Resources | None = None
        try:
            client = Langfuse(tracer_provider=provider)
            owned = client._resources
            if owned is not None:
                if owned.tracer_provider is not None and owned.tracer_provider is not provider:
                    client = None  # A pre-existing client belongs to its original owner.
                    raise _core_error(RuntimeError, "Langfuse provider already owned")
                # The SDK's unbounded atexit callback must not outlive our bounded one.
                atexit.unregister(owned.shutdown)
                atexit.unregister(owned.prompt_cache._task_manager.shutdown)
            instrumentor.instrument(
                tracer_provider=provider,
                exclusive_processor=True,
                config=TraceConfig(
                    hide_inputs=True,
                    hide_outputs=True,
                    hide_llm_invocation_parameters=True,
                    hide_llm_tools=True,
                ),
            )
            resources = _Resources(client, provider, instrumentor)
            resources.project_worker = threading.Thread(
                target=_resolve_project,
                args=(resources,),
                name="swarm-arena-tracing-project",
                daemon=True,
            )
            resources.project_worker.start()
        except BaseException as error:
            if client is not None:
                _closing = resources or _Resources(client, provider, instrumentor)
                _close(_closing, _EXIT_TIMEOUT_S)
                if not _closing.worker.is_alive():
                    _closing = None
            else:
                provider.shutdown()
            raise _core_error(
                RuntimeError, f"Langfuse setup failed: {_exception_type_name(error)}"
            ) from None
        _resources = resources
        atexit.register(_close_at_exit)


@contextmanager
def _guarded_context(factory: Callable[[], AbstractContextManager[Any]]) -> Generator[None]:
    """Contain tracing failures without replacing a locally escaping outcome."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        outcome = _Outcome(capture_caller=False)
    else:
        outcome = _Outcome()
    try:
        manager = factory()
        manager.__enter__()
    except BaseException as error:
        outcome.contain(error, RuntimeError, "Langfuse context")
    failure: BaseException | None = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            manager.__exit__(
                type(failure) if failure is not None else None,
                failure,
                failure.__traceback__ if failure is not None else None,
            )
        except BaseException as error:
            if failure is None:
                outcome.propagate_control(error)
            _log.warning("%s", _exception_type_name(error))


@contextmanager
def component_span(scope: LLMScope, *, candidate_id: str | None = None) -> Generator[None]:
    """Propagate run/session attributes through a component's owned lifetime.

    Parameters
    ----------
    scope : LLMScope
        Active component consent and identifiers, inherited by child scopes.
    candidate_id : str or None, optional
        Candidate identifier associated with this component.

    Yields
    ------
    None
        Attributes are active until all owned actions and accounting finish.

    Examples
    --------
    A disabled scope needs no Langfuse installation or initialized client:

    >>> from unittest.mock import Mock
    >>> scope = Mock(tracing=TracingConfig())
    >>> with component_span(scope):
    ...     assert current_trace_url() is None
    """
    if not scope.tracing.langfuse:
        yield
        return
    from langfuse import propagate_attributes

    with _guarded_context(
        lambda: propagate_attributes(
            session_id=scope.run_id,
            metadata={
                "goal": scope.goal_id,
                "team": scope.team,
                "agent": scope.agent,
                "round": str(scope.round),
                "candidate_id": candidate_id or "",
            },
            tags=[f"role:{scope.role}", f"goal:{scope.goal_id}"],
        )
    ):
        yield


@contextmanager
def _call_trace(scope: LLMScope) -> Generator[None]:
    """Give direct model calls both an SDK trace and a current export span."""
    if not scope.tracing.langfuse:
        yield
        return
    from agents import custom_span, trace

    with (
        _guarded_context(
            lambda: trace(workflow_name=f"swarm-arena {scope.role}", group_id=scope.run_id)
        ),
        _guarded_context(lambda: custom_span(name=f"swarm-arena {scope.role} call")),
    ):
        yield


def current_trace_url() -> str | None:
    """Return the current consenting scope's Langfuse URL without network I/O.

    Returns
    -------
    str or None
        Trace URL, or None when consent, a current trace, or resolved project
        metadata is absent. Failed project resolution is not retried.

    Examples
    --------
    >>> current_trace_url() is None
    True
    """
    scope = current_llm_scope()
    resources = _resources
    if scope is None or not scope.tracing.langfuse or resources is None:
        return None
    try:
        trace_id = resources.client.get_current_trace_id()
        if not trace_id:
            return None
        prefix = resources.project_url
        return prefix + trace_id if prefix is not None and type(trace_id) is str else None
    except BaseException:
        return None
