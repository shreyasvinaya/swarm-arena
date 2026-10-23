"""Offline tracing consent, SDK export boundaries and process-lifetime cleanup."""

import builtins
import threading
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
from unittest.mock import Mock
from uuid import uuid4

import pytest
from _fakes.cleanup import observe_task, settle_tasks

pytest.importorskip("agents")

# isort: split
import agents
import anyio
from _fakes.llm_run import open_llm_run
from _fakes.secrecy import assert_secret_absent
from agents.models.interface import ModelTracing
from agents.tracing import TracingProcessor
from llm.test_judge import candidates, configuration, turn
from llm.test_manager import config as manager_config
from llm.test_manager import context, finish

from swarm_arena import Goal
from swarm_arena._budget.llm_scope import llm_scope
from swarm_arena.config import TracingConfig
from swarm_arena.llm import AgentsSDKManager, LLMJudge, _langfuse
from swarm_arena.scoring import JudgeContext
from swarm_arena.testing import FakeModel


class Spy(TracingProcessor):
    def __init__(self) -> None:
        self.traces: list[Any] = []
        self.spans: list[Any] = []

    def on_trace_start(self, trace: Any) -> None:
        self.traces.append(trace)

    def on_trace_end(self, trace: Any) -> None:
        pass

    def on_span_start(self, span: Any) -> None:
        self.spans.append(span)

    def on_span_end(self, span: Any) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    langfuse = pytest.importorskip("langfuse")
    import httpx
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    original = langfuse.Langfuse
    clients: list[Any] = []

    def project_http(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/projects"
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "test",
                        "name": "test",
                        "metadata": {},
                        "organization": {"id": "org", "name": "test"},
                    }
                ]
            },
        )

    http = httpx.Client(transport=httpx.MockTransport(project_http))

    def client(**kwargs: Any) -> Any:
        result = original(
            **kwargs,
            public_key="pk-test-" + uuid4().hex,
            secret_key="sk-test",
            base_url="https://langfuse.example",
            span_exporter=exporter,
            httpx_client=http,
        )
        clients.append(result)
        return result

    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")
    monkeypatch.setattr(langfuse, "Langfuse", client)
    try:
        yield exporter, clients
    finally:
        resources = _langfuse._resources or _langfuse._closing
        if resources is not None and resources.project_worker is not None:
            resources.project_worker.join(60)
            assert not resources.project_worker.is_alive()
        _langfuse._reset_langfuse(60)
        http.close()


def wait_project_resolution() -> None:
    resources = _langfuse._resources
    assert resources is not None
    assert resources.project_worker is not None
    resources.project_worker.join(60)
    assert not resources.project_worker.is_alive()


@contextmanager
def spy_processor() -> Generator[Spy]:
    from agents.tracing import setup
    from agents.tracing.provider import DefaultTraceProvider

    _langfuse.setup_langfuse(TracingConfig())
    provider = setup.GLOBAL_TRACE_PROVIDER
    assert isinstance(provider, DefaultTraceProvider)
    previous = provider._multi_processor._processors
    spy = Spy()
    agents.add_trace_processor(spy)
    try:
        yield spy
    finally:
        provider.set_processors(list(previous))


def test_setup_does_nothing_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    original = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in {"langfuse", "openinference"}:
            pytest.fail("optional tracing dependency imported while disabled")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    _langfuse.setup_langfuse(TracingConfig())
    assert _langfuse.current_trace_url() is None


def test_missing_extra_is_actionable_and_value_free(monkeypatch: pytest.MonkeyPatch) -> None:
    from swarm_arena._errors import ConfigError

    original = builtins.__import__

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse":
            raise ImportError("SENTINEL_SECRET")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ConfigError, match=r"swarm-arena\[langfuse\]") as caught:
        _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    assert (
        'uv add "swarm-arena[langfuse] @ git+https://github.com/shreyasvinaya/swarm-arena"'
        in str(caught.value)
    )
    assert_secret_absent(caught.value, "SENTINEL_SECRET")


def test_setup_instruments_once(exported: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    calls: list[dict[str, Any]] = []
    original = OpenAIAgentsInstrumentor.instrument

    def instrument(self: Any, **kwargs: Any) -> None:
        calls.append(kwargs)
        original(self, **kwargs)

    monkeypatch.setattr(OpenAIAgentsInstrumentor, "instrument", instrument)
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    _langfuse.setup_langfuse(TracingConfig())
    assert len(calls) == len(exported[1]) == 1
    assert calls[0]["exclusive_processor"] is True
    assert calls[0]["tracer_provider"] is exported[1][0]._resources.tracer_provider


async def judge_call(tmp_path: Path, enabled: bool) -> tuple[Any, list[Any]]:
    fake = FakeModel([turn(1)])
    fake.register("a")
    cfg = configuration().model_copy(update={"tracing": TracingConfig(langfuse=enabled)})
    goal = Goal(id="g", namespace="n")
    async with open_llm_run(tmp_path, config=cfg) as run:
        if enabled:
            wait_project_resolution()
        with llm_scope(run.scope("judge")):
            await LLMJudge("rubric", ["a"]).judge(
                goal,
                candidates(1),
                JudgeContext(goal=goal, executor=Mock(), artifact_root=str(tmp_path)),
            )
        return fake, run.calls()


async def manager_call(tmp_path: Path, enabled: bool) -> tuple[Any, list[Any]]:
    fake = FakeModel([finish()])
    fake.register("opus")
    async with open_llm_run(
        tmp_path, config=manager_config(tracing=TracingConfig(langfuse=enabled))
    ) as run:
        if enabled:
            wait_project_resolution()
        with llm_scope(run.scope("manager")):
            await AgentsSDKManager(model="opus").run_round(context())
        return fake, run.calls()


async def test_no_spans_leave_the_process_when_langfuse_is_off(tmp_path: Path) -> None:
    with spy_processor() as spy:
        for call in (manager_call, judge_call):
            fake, recorded = await call(tmp_path, False)
            assert fake.calls[0]["tracing"] is ModelTracing.DISABLED
            assert recorded[0].trace_url is None
        assert spy.traces == []
        assert spy.spans == []


async def test_component_span_attributes(
    tmp_path: Path, exported: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import langfuse

    seen: list[dict[str, Any]] = []
    original = langfuse.propagate_attributes

    @contextmanager
    def propagate(**kwargs: Any) -> Generator[None]:
        seen.append(kwargs)
        with original(**kwargs):
            yield

    monkeypatch.setattr(langfuse, "propagate_attributes", propagate)
    async with open_llm_run(
        tmp_path, config=manager_config(tracing=TracingConfig(langfuse=True))
    ) as run:
        scope = run.scope("manager")
        with llm_scope(scope), _langfuse.component_span(scope):
            pass
        assert seen == [
            {
                "session_id": run.run_id,
                "metadata": {
                    "goal": "g",
                    "team": "t",
                    "agent": "a",
                    "round": "1",
                    "candidate_id": "",
                },
                "tags": ["role:manager", "goal:g"],
            }
        ]


async def test_trace_url_reaches_the_llm_call(tmp_path: Path, exported: Any) -> None:
    _, calls = await manager_call(tmp_path, True)
    client = exported[1][0]
    client.flush()
    spans = exported[0].get_finished_spans()
    assert spans
    assert (
        calls[0].trace_url
        == f"https://langfuse.example/project/test/traces/{spans[0].context.trace_id:032x}"
    )
    assert any(s.attributes.get("session.id") == calls[0].run_id for s in spans)


async def test_judge_call_opens_an_sdk_trace_and_sets_trace_url(
    tmp_path: Path, exported: Any
) -> None:
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    with spy_processor() as spy:
        fake, calls = await judge_call(tmp_path, True)
        exported[1][0].flush()
        spans = exported[0].get_finished_spans()
        assert [t.name for t in spy.traces] == ["swarm-arena judge"]
        assert fake.calls[0]["tracing"] is ModelTracing.ENABLED_WITHOUT_DATA
        assert (
            calls[0].trace_url
            == f"https://langfuse.example/project/test/traces/{spans[0].context.trace_id:032x}"
        )


async def test_disabled_after_enabled_exports_nothing(tmp_path: Path, exported: Any) -> None:
    await manager_call(tmp_path, True)
    exported[1][0].flush()
    exported[0].clear()
    with spy_processor() as spy:
        await manager_call(tmp_path, False)
        await judge_call(tmp_path, False)
        exported[1][0].flush()
        assert exported[0].get_finished_spans() == ()
        assert spy.traces == []
        assert spy.spans == []


async def test_concurrent_mixed_scopes_export_only_consenting_run(
    tmp_path: Path, exported: Any
) -> None:
    from swarm_arena.llm._structured import structured_call

    FakeModel([turn(1), turn(1)]).register("a")
    cfg = configuration().model_copy(update={"tracing": TracingConfig(langfuse=True)})
    async with open_llm_run(tmp_path, config=cfg) as run:
        wait_project_resolution()
        enabled = run.scope("judge", agent="enabled")
        disabled = replace(run.scope("judge", agent="disabled"), tracing=TracingConfig())
        entered = anyio.Event()
        release = anyio.Event()

        async def call(scope: Any) -> None:
            with llm_scope(scope), _langfuse.component_span(scope):
                if scope.tracing.langfuse:
                    entered.set()
                    await release.wait()
                else:
                    await entered.wait()
                await structured_call(
                    alias="a",
                    role="judge",
                    models=scope.models,
                    namespace="n",
                    system="s",
                    user="u",
                    schema={"type": "object"},
                    name="result",
                    no_cache_read=True,
                    temperature=0,
                )
                release.set()

        with anyio.fail_after(60):
            async with anyio.create_task_group() as group:
                group.start_soon(call, enabled)
                group.start_soon(call, disabled)
        calls = run.calls()
        assert {c.agent: c.trace_url is not None for c in calls} == {
            "enabled": True,
            "disabled": False,
        }
    exported[1][0].flush()
    spans = exported[0].get_finished_spans()
    assert spans
    assert all(s.attributes.get("langfuse.trace.metadata.agent") == '"enabled"' for s in spans)


async def test_trace_url_is_none_without_a_current_trace(
    tmp_path: Path, exported: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = manager_config(tracing=TracingConfig(langfuse=True))
    async with open_llm_run(tmp_path, config=cfg) as run:
        client = exported[1][0]
        current = Mock(wraps=client.get_current_trace_id)
        monkeypatch.setattr(client, "get_current_trace_id", current)
        with llm_scope(run.scope("manager")):
            assert _langfuse.current_trace_url() is None
        current.assert_called_once_with()


async def test_trace_id_failure_is_ignored(
    tmp_path: Path, exported: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    monkeypatch.setattr(
        exported[1][0], "get_current_trace_id", Mock(side_effect=RuntimeError("SENTINEL_SECRET"))
    )
    _, calls = await manager_call(tmp_path, True)
    assert calls[0].trace_url is None


def test_exit_close_is_bounded_when_exporter_blocks(
    exported: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    assert _langfuse._EXIT_TIMEOUT_S == 5
    entered, release, returned, finished = (threading.Event() for _ in range(4))
    exporter = exported[0]
    original = exporter.shutdown

    def shutdown() -> None:
        entered.set()
        try:
            assert release.wait(300)
            original()
        finally:
            finished.set()

    monkeypatch.setattr(exporter, "shutdown", shutdown)
    monkeypatch.setattr(_langfuse, "_EXIT_TIMEOUT_S", 0.05)

    def close() -> None:
        _langfuse._close_at_exit()
        returned.set()

    owner = threading.Thread(target=close)
    owner.start()
    try:
        assert entered.wait(60)
        assert returned.wait(30), "exit waited for the blocked exporter"
        assert not finished.is_set()
        assert "TimeoutError" in caplog.text
    finally:
        release.set()
        owner.join(60)
        assert finished.wait(60)
        assert not owner.is_alive()


def test_cleanup_errors_log_only_type(
    exported: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    client = exported[1][0]
    monkeypatch.setattr(client, "flush", Mock(side_effect=ValueError("SENTINEL_SECRET")))
    _langfuse._reset_langfuse(60)
    assert "ValueError" in caplog.text
    assert "SENTINEL_SECRET" not in caplog.text


async def test_component_cleanup_cannot_replace_body_control(
    tmp_path: Path, exported: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import langfuse

    from swarm_arena._errors import _stop_cleanup_error

    @contextmanager
    def broken(**kwargs: Any) -> Generator[None]:
        try:
            yield
        finally:
            raise ValueError("SENTINEL_SECRET")

    monkeypatch.setattr(langfuse, "propagate_attributes", broken)
    async with open_llm_run(
        tmp_path, config=manager_config(tracing=TracingConfig(langfuse=True))
    ) as run:
        scope = run.scope("manager")
        control = _stop_cleanup_error()
        with (
            llm_scope(scope),
            pytest.raises(RuntimeError) as caught,
            _langfuse.component_span(scope),
        ):
            raise control
        assert caught.value is control
        assert_secret_absent(caught.value, "SENTINEL_SECRET")


async def test_success_inside_unrelated_except_still_checks_trace_cleanup(
    tmp_path: Path, exported: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import langfuse

    from swarm_arena.manager import RoundReport

    @contextmanager
    def broken(**kwargs: Any) -> Generator[None]:
        yield
        raise ValueError("SENTINEL_SECRET")

    monkeypatch.setattr(langfuse, "propagate_attributes", broken)
    FakeModel([finish()]).register("opus")
    async with open_llm_run(
        tmp_path, config=manager_config(tracing=TracingConfig(langfuse=True))
    ) as run:
        try:
            raise RuntimeError("unrelated")
        except RuntimeError:
            with llm_scope(run.scope("manager")):
                report = await AgentsSDKManager(model="opus").run_round(context())
        assert isinstance(report, RoundReport)
        assert len(run.calls()) == 1
        assert "ValueError" in caplog.text
        assert "SENTINEL_SECRET" not in caplog.text


async def test_context_construction_failure_is_value_free(
    tmp_path: Path, exported: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import langfuse

    monkeypatch.setattr(
        langfuse, "propagate_attributes", Mock(side_effect=ValueError("SENTINEL_SECRET"))
    )
    async with open_llm_run(
        tmp_path, config=manager_config(tracing=TracingConfig(langfuse=True))
    ) as run:
        with pytest.raises(RuntimeError) as caught, _langfuse.component_span(run.scope("manager")):
            pass
        assert_secret_absent(caught.value, "SENTINEL_SECRET")


def test_no_scope_exports_nothing(exported: Any) -> None:
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    with agents.trace("unrelated"), agents.custom_span("unrelated"):
        assert _langfuse.current_trace_url() is None
    exported[1][0].flush()
    assert exported[0].get_finished_spans() == ()


def test_reset_allows_same_key_reinitialization(
    exported: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from uuid import UUID

    import langfuse

    monkeypatch.setitem(globals(), "uuid4", lambda: UUID(int=1))
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    first = exported[1][0]._resources
    _langfuse._reset_langfuse(60)
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    second = exported[1][1]._resources
    assert second is not first
    assert not second._shutdown
    assert callable(langfuse.get_client)


async def test_planner_and_curator_trace_their_child_calls(tmp_path: Path, exported: Any) -> None:
    import json

    from _fakes.planning_llm import planner_config, planner_output
    from llm.test_curator import configuration as curator_config
    from llm.test_curator import group, learning, view
    from llm.test_curator import turn as curator_turn

    from swarm_arena.llm import LLMCurator, LLMPlanner
    from swarm_arena.testing import FakeTurn

    fake = FakeModel([FakeTurn(text=json.dumps(planner_output()))])
    fake.register("planner")
    cfg = planner_config(tmp_path).model_copy(update={"tracing": TracingConfig(langfuse=True)})
    async with open_llm_run(tmp_path, config=cfg) as run:
        wait_project_resolution()
        scope = run.scope("planner")
        with llm_scope(scope):
            await LLMPlanner("planner")._draft_at(
                Goal(id="g", namespace="n"), cfg, None, base_dir=tmp_path
            )
        assert scope.call_ids == [c.id for c in run.calls()]
        assert run.calls()[0].trace_url
    fake = FakeModel([curator_turn([group([0, 1])])])
    fake.register("cur")
    cfg = curator_config().model_copy(update={"tracing": TracingConfig(langfuse=True)})
    async with open_llm_run(tmp_path, config=cfg) as run:
        scope = run.scope("curator")
        with llm_scope(scope):
            await LLMCurator(model="cur").curate("physika", [learning(1), learning(2)], view())
        assert scope.call_ids == [c.id for c in run.calls()]
        assert run.calls()[0].trace_url
    exported[1][0].flush()
    names = {s.name for s in exported[0].get_finished_spans()}
    assert {"swarm-arena planner", "swarm-arena curator"} <= names


@pytest.mark.parametrize("kind", ["caller", "fenced", "terminal"])
async def test_failing_trace_exit_preserves_control(
    tmp_path: Path, exported: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import asyncio

    import langfuse

    from swarm_arena._errors import Fenced, _core_error, _stop_cleanup_error

    @contextmanager
    def broken(**kwargs: Any) -> Generator[None]:
        try:
            yield
        finally:
            raise ValueError("SENTINEL_SECRET")

    monkeypatch.setattr(langfuse, "propagate_attributes", broken)
    async with open_llm_run(
        tmp_path, config=manager_config(tracing=TracingConfig(langfuse=True))
    ) as run:
        entered = anyio.Event()
        error = _core_error(Fenced, "ownership lost") if kind == "fenced" else _stop_cleanup_error()

        async def body() -> None:
            with llm_scope(run.scope("manager")), _langfuse.component_span(run.scopes[-1]):
                entered.set()
                if kind == "caller":
                    await anyio.sleep_forever()
                raise error

        task = asyncio.create_task(body())
        try:
            with anyio.fail_after(60):
                await entered.wait()
                if kind == "caller":
                    task.cancel()
                with pytest.raises(
                    asyncio.CancelledError if kind == "caller" else type(error)
                ) as caught:
                    await observe_task(task)
                assert_secret_absent(caught.value, "SENTINEL_SECRET")
                if kind != "caller":
                    assert caught.value is error
        finally:
            task.cancel()
            await settle_tasks(task)


def test_registers_one_bounded_exit_owner(exported: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import atexit

    _langfuse.setup_langfuse(TracingConfig())
    callbacks: list[Any] = []
    register = atexit.register
    unregister = atexit.unregister

    def add(function: Any, *args: Any, **kwargs: Any) -> Any:
        callbacks.append(function)
        return register(function, *args, **kwargs)

    def remove(function: Any) -> None:
        callbacks[:] = [callback for callback in callbacks if callback != function]
        unregister(function)

    monkeypatch.setattr(atexit, "register", add)
    monkeypatch.setattr(atexit, "unregister", remove)
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    assert callbacks == [_langfuse._close_at_exit]
    _langfuse._reset_langfuse(60)
    assert callbacks == []


def test_setup_failure_is_value_free(exported: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

    monkeypatch.setattr(
        OpenAIAgentsInstrumentor, "instrument", Mock(side_effect=ValueError("SENTINEL_SECRET"))
    )
    with pytest.raises(RuntimeError, match="Langfuse setup failed: ValueError") as caught:
        _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    assert_secret_absent(caught.value, "SENTINEL_SECRET")
    assert exported[1][0]._resources._shutdown


async def test_two_stage_calls_export_sdk_generation_spans(
    tmp_path: Path, exported: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agents.models.interface import ModelResponse

    from swarm_arena.llm import _structured
    from swarm_arena.testing import FakeTurn

    class GeneratingFake(FakeModel):
        async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
            with agents.generation_span(model="fake", disabled=args[6].is_disabled()):
                return await super().get_response(*args, **kwargs)

    fake = GeneratingFake([FakeTurn(text="draft"), turn(1)])
    fake.register("a")
    monkeypatch.setattr(_structured, "supports", lambda spec, feature: False)
    cfg = configuration().model_copy(update={"tracing": TracingConfig(langfuse=True)})
    async with open_llm_run(tmp_path, config=cfg) as run:
        wait_project_resolution()
        with llm_scope(run.scope("judge")):
            await _structured.structured_call(
                alias="a",
                role="judge",
                models=cfg.models,
                namespace="n",
                system="s",
                user="u",
                schema={"type": "object"},
                name="result",
                no_cache_read=True,
                temperature=0,
            )
        assert len(run.calls()) == 2
        assert all(c.trace_url for c in run.calls())
    exported[1][0].flush()
    spans = exported[0].get_finished_spans()
    generations = [s for s in spans if s.attributes.get("openinference.span.kind") == "LLM"]
    assert len(generations) == 2
    assert {f"{s.context.trace_id:032x}" for s in generations} == {
        c.trace_url.rsplit("/", 1)[1] for c in run.calls() if c.trace_url is not None
    }


def test_exit_close_does_not_start_threads(exported: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    # Python 3.12+ forbids starting new threads from an atexit handler.
    monkeypatch.setattr(
        threading.Thread,
        "start",
        Mock(side_effect=RuntimeError("can't create new thread at interpreter shutdown")),
    )
    _langfuse._close_at_exit()
    assert exported[1][0]._resources._shutdown


def test_exit_close_bounds_owner_lock_wait(
    exported: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    monkeypatch.setattr(_langfuse, "_EXIT_TIMEOUT_S", 0.05)
    returned = threading.Event()

    def close() -> None:
        _langfuse._close_at_exit()
        returned.set()

    owner = threading.Thread(target=close)
    _langfuse._lock.acquire()
    try:
        owner.start()
        assert returned.wait(30), "exit waited indefinitely for tracing setup"
        assert "TimeoutError" in caplog.text
    finally:
        _langfuse._lock.release()
        owner.join(60)
        assert not owner.is_alive()


@pytest.mark.parametrize("blocked", [False, True])
def test_actual_process_exit_uses_bounded_daemon(tmp_path: Path, blocked: bool) -> None:
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import atexit
        import threading
        from agents.tracing import processors
        from swarm_arena._model.settings import TracingConfig
        from swarm_arena.llm import _langfuse

        def reject_default_processor() -> None:
            raise AssertionError("default vendor processor created")

        processors.default_processor = reject_default_processor

        _langfuse.setup_langfuse(TracingConfig())
        provider_entered = threading.Event()

        class Client:
            _resources = None
            def flush(self):
                print("flushed", flush=True)
            def shutdown(self):
                print("client closed", flush=True)

        class Instrumentor:
            is_instrumented_by_opentelemetry = False

        class Provider:
            def shutdown(self):
                print("provider entered", flush=True)
                provider_entered.set()
                if BLOCKED:
                    threading.Event().wait(300)
                print("provider closed", flush=True)

        resources = _langfuse._Resources(Client(), Provider(), Instrumentor())
        _langfuse._resources = resources
        if BLOCKED:
            # Finish ordinary cleanup before applying the short exit deadline.
            resources.requested.set()
            assert provider_entered.wait(60), "provider shutdown never became ready"
        _langfuse._EXIT_TIMEOUT_S = 0.05 if BLOCKED else 60
        atexit.register(_langfuse._close_at_exit)
    """).replace("BLOCKED", repr(blocked))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        # Allow cold imports, but fail before the held shutdown's safety release.
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "flushed" in result.stdout
    assert "client closed" in result.stdout
    assert "provider entered" in result.stdout
    assert ("provider closed" in result.stdout) is not blocked
    assert result.stderr.strip() == ("TimeoutError" if blocked else "")


@pytest.mark.parametrize("enabled", [False, True])
async def test_caller_cannot_forge_trace_url(tmp_path: Path, exported: Any, enabled: bool) -> None:
    from llm.test_hooks import config, invoke, metered

    from swarm_arena.testing import FakeTurn

    capture = FakeModel([FakeTurn(text="ok")])
    cfg = config(tracing=TracingConfig(langfuse=enabled))
    async with open_llm_run(tmp_path, config=cfg) as run:
        with llm_scope(run.scope("manager")):
            await invoke(metered(capture), agents.ModelSettings(metadata={"trace_url": "forged"}))
        assert run.calls()[0].trace_url is None
        assert "trace_url" not in capture.calls[0]["model_settings"].metadata


def test_off_setup_never_constructs_default_vendor_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents.tracing import processors, setup

    monkeypatch.setattr(setup, "GLOBAL_TRACE_PROVIDER", None)
    monkeypatch.setattr(
        processors,
        "default_processor",
        Mock(side_effect=AssertionError("default exporter created")),
    )
    _langfuse.setup_langfuse(TracingConfig())
    with agents.trace("disabled", disabled=True):
        pass


def test_existing_instrumentor_is_not_silently_reused(exported: Any) -> None:
    from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    instrumentor = OpenAIAgentsInstrumentor()
    provider = TracerProvider(shutdown_on_exit=False)
    instrumentor.instrument(tracer_provider=provider)
    try:
        with pytest.raises(RuntimeError, match="instrumentor already owned"):
            _langfuse.setup_langfuse(TracingConfig(langfuse=True))
        assert exported[1] == []
        assert instrumentor.is_instrumented_by_opentelemetry
    finally:
        instrumentor.uninstrument()
        provider.shutdown()
        agents.set_trace_processors([])


@pytest.mark.parametrize("response", ["success", "timeout", "rate_limit", "empty"])
async def test_project_resolution_never_blocks_llm_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    response: str,
) -> None:
    import httpx
    import langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    entered, release = threading.Event(), threading.Event()
    requests: list[httpx.Request] = []
    clients: list[Any] = []
    original = langfuse.Langfuse

    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/public/projects"
        requests.append(request)
        entered.set()
        assert release.wait(300), "project lookup was not released"
        if response == "timeout":
            raise httpx.ConnectTimeout("SENTINEL_SECRET")
        if response == "rate_limit":
            return httpx.Response(429, headers={"Retry-After": "60"}, json={})
        data = (
            []
            if response == "empty"
            else [
                {
                    "id": "resolved",
                    "name": "test",
                    "metadata": {},
                    "organization": {"id": "org", "name": "test"},
                }
            ]
        )
        return httpx.Response(200, json={"data": data})

    http = httpx.Client(transport=httpx.MockTransport(transport))

    def client(**kwargs: Any) -> Any:
        result = original(
            **kwargs,
            public_key="pk-test-" + uuid4().hex,
            secret_key="sk-test",
            base_url="https://langfuse.example",
            span_exporter=InMemorySpanExporter(),
            httpx_client=http,
        )
        clients.append(result)
        return result

    monkeypatch.setenv("LANGFUSE_MEDIA_UPLOAD_ENABLED", "false")
    monkeypatch.setattr(langfuse, "Langfuse", client)
    cfg = manager_config(tracing=TracingConfig(langfuse=True))
    resources: Any = None
    try:
        _langfuse.setup_langfuse(TracingConfig(langfuse=True))
        resources = _langfuse._resources
        assert resources is not None
        assert resources.project_worker is not None
        assert await anyio.to_thread.run_sync(entered.wait, 60)
        assert clients[0]._project_id is None
        get_url = Mock(side_effect=AssertionError("network-capable URL helper called"))
        monkeypatch.setattr(clients[0], "get_trace_url", get_url)
        FakeModel([finish()]).register("opus")
        # The transport remains held until this complete, journaled round returns.
        with anyio.fail_after(60):
            async with open_llm_run(tmp_path, config=cfg) as run:
                with llm_scope(run.scope("manager")):
                    await AgentsSDKManager(model="opus").run_round(context())
                assert len(run.calls()) == 1
                assert run.calls()[0].trace_url is None
        assert not release.is_set()
        release.set()
        await anyio.to_thread.run_sync(resources.project_worker.join, 60)
        assert not resources.project_worker.is_alive(), "project lookup retried or did not finish"
        _, calls = await manager_call(tmp_path, True)
        assert (calls[0].trace_url is not None) is (response == "success")
        get_url.assert_not_called()
        assert len(requests) == 1
        assert all(0 < value <= 5 for value in requests[0].extensions["timeout"].values())
        assert "SENTINEL_SECRET" not in caplog.text
        if response != "success":
            assert any(record.name == _langfuse.__name__ for record in caplog.records)
    finally:
        release.set()
        if resources is not None and resources.project_worker is not None:
            await anyio.to_thread.run_sync(resources.project_worker.join, 60)
            assert not resources.project_worker.is_alive()
        _langfuse._reset_langfuse(60)
        await anyio.to_thread.run_sync(http.close)


def test_close_does_not_create_a_missing_sdk_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from agents.tracing import processors, setup

    monkeypatch.setattr(setup, "GLOBAL_TRACE_PROVIDER", None)
    default = Mock(side_effect=AssertionError("default vendor processor created"))
    monkeypatch.setattr(processors, "default_processor", default)
    client = Mock(_resources=None)
    resources = _langfuse._Resources(client, Mock(), Mock(is_instrumented_by_opentelemetry=False))
    _langfuse._close(resources, 60)
    assert not resources.worker.is_alive()
    default.assert_not_called()
    assert setup.GLOBAL_TRACE_PROVIDER is None
    client.flush.assert_called_once_with()


def test_reset_refuses_new_owner_until_cleanup_finishes(
    exported: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    resources = _langfuse._resources
    assert resources is not None
    entered, release = threading.Event(), threading.Event()
    original = resources.instrumentor.uninstrument

    def held() -> None:
        entered.set()
        assert release.wait(300)
        original()

    monkeypatch.setattr(resources.instrumentor, "uninstrument", held)
    resources.requested.set()
    try:
        assert entered.wait(60)
        _langfuse._reset_langfuse(0.05)
        with pytest.raises(RuntimeError, match="Langfuse cleanup pending"):
            _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    finally:
        release.set()
        resources.worker.join(60)
        assert not resources.worker.is_alive()
    _langfuse.setup_langfuse(TracingConfig(langfuse=True))
    assert _langfuse._resources is not resources
    assert resources.instrumentor.is_instrumented_by_opentelemetry


async def test_missing_extra_hint_survives_swarm_run(
    run_config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from _fakes.run import plan

    from swarm_arena import Swarm
    from swarm_arena._errors import ConfigError

    original = builtins.__import__

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langfuse":
            raise ImportError("SENTINEL_SECRET")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    cfg = run_config.model_copy(update={"tracing": TracingConfig(langfuse=True)})
    with pytest.raises(ConfigError, match=r"swarm-arena\[langfuse\]") as caught:
        await Swarm(cfg).run(plan(rounds=1), run_id="missing-tracing-extra")
    assert_secret_absent(caught.value, "SENTINEL_SECRET")


@pytest.mark.parametrize("kind", ["fenced", "terminal"])
async def test_trace_exit_control_still_wins_after_success(
    tmp_path: Path, exported: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import langfuse

    from swarm_arena._errors import Fenced, _core_error, _stop_cleanup_error

    error = _core_error(Fenced, "ownership lost") if kind == "fenced" else _stop_cleanup_error()

    @contextmanager
    def broken(**kwargs: Any) -> Generator[None]:
        yield
        raise error

    monkeypatch.setattr(langfuse, "propagate_attributes", broken)
    async with open_llm_run(
        tmp_path, config=manager_config(tracing=TracingConfig(langfuse=True))
    ) as run:
        with (
            llm_scope(run.scope("manager")),
            pytest.raises(type(error), match=str(error)),
            _langfuse.component_span(run.scopes[-1]),
        ):
            pass


@pytest.mark.parametrize("role", ["manager", "judge"])
async def test_provider_error_spans_are_value_free(
    tmp_path: Path,
    exported: Any,
    monkeypatch: pytest.MonkeyPatch,
    role: Literal["manager", "judge"],
) -> None:
    import json
    from unittest.mock import AsyncMock

    import litellm
    from _fakes.llm_components import missing_callback_timeout
    from _fakes.run import all_events, plan
    from agents.extensions.models.litellm_model import LitellmModel
    from langfuse._client.span_processor import LangfuseSpanProcessor
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    from swarm_arena import Limits, Ref, RunConfig, Swarm
    from swarm_arena.config import ModelCatalog, ModelSpec

    missing_callback_timeout(monkeypatch)
    secret = "PROVIDER_ERROR_SECRET_8c031"
    received: list[str] = []
    original_start = LangfuseSpanProcessor.on_start
    original_end = LangfuseSpanProcessor.on_end

    def on_start(self: Any, span: Any, parent_context: Any = None) -> None:
        received.append(json.dumps(dict(span.attributes or {})))
        original_start(self, span, parent_context)

    def on_end(self: Any, span: Any) -> None:
        received.append(
            json.dumps(
                {
                    "status": span.status.description,
                    "attributes": dict(span.attributes or {}),
                    "events": [
                        {"name": event.name, "attributes": dict(event.attributes or {})}
                        for event in span.events
                    ],
                }
            )
        )
        original_end(self, span)

    monkeypatch.setattr(LangfuseSpanProcessor, "on_start", on_start)
    monkeypatch.setattr(LangfuseSpanProcessor, "on_end", on_end)
    modes: list[ModelTracing] = []
    runner_sensitive: list[bool] = []
    original_response = LitellmModel.get_response
    original_run = agents.Runner.run

    async def response(self: Any, *args: Any, **kwargs: Any) -> Any:
        modes.append(args[6])
        return await original_response(self, *args, **kwargs)

    async def runner(*args: Any, **kwargs: Any) -> Any:
        runner_sensitive.append(kwargs["run_config"].trace_include_sensitive_data)
        return await original_run(*args, **kwargs)

    monkeypatch.setattr(LitellmModel, "get_response", response)
    monkeypatch.setattr(agents.Runner, "run", runner)

    async def fail_provider(**kwargs: Any) -> Any:
        error = RuntimeError(secret)
        span = trace.get_current_span()
        span.set_status(Status(StatusCode.ERROR, secret))
        span.set_attribute("error.message", secret)
        span.set_attribute("exception.stacktrace", secret)
        span.record_exception(error, attributes={"provider.detail": secret})
        resources = _langfuse._resources
        assert resources is not None
        tracer = resources.provider.get_tracer("openinference.instrumentation.openai_agents")
        with tracer.start_as_current_span(
            "provider failure",
            attributes={"error.message": secret, "openinference.span.kind": "LLM"},
        ):
            raise error

    dispatch = AsyncMock(side_effect=fail_provider)
    monkeypatch.setattr(litellm, "acompletion", dispatch)
    proposed = plan(rounds=1)
    if role == "manager":
        proposed.teams[0].agents[0] = (
            proposed.teams[0]
            .agents[0]
            .model_copy(update={"manager": Ref.of(AgentsSDKManager(model="m"))})
        )
    else:
        proposed.judges.append(Ref.of(LLMJudge("rubric", ["m"])))
    cfg = RunConfig(
        run_root=str(tmp_path / "runs"),
        arena_root=str(tmp_path / "arena"),
        artifact_root=str(tmp_path / "artifacts"),
        limits=Limits(max_tokens=1000000),
        tracing=TracingConfig(langfuse=True),
        models=ModelCatalog(
            {
                "m": ModelSpec(
                    model="openai/gpt-4o-mini",
                    roles=[role],
                    max_tokens=100,
                    prices=ModelSpec.Prices(input=1, output=1),
                )
            }
        ),
    )
    with anyio.fail_after(30):
        await Swarm(cfg).run(proposed)
    assert dispatch.await_count == 1
    assert modes == [ModelTracing.ENABLED_WITHOUT_DATA]
    assert runner_sensitive == ([False] if role == "manager" else [])
    exported[1][0].flush()
    spans = exported[0].get_finished_spans()
    assert spans
    errors = [span for span in spans if span.status.status_code == StatusCode.ERROR]
    assert errors
    assert all(span.status.description in {"Exception", "RuntimeError"} for span in errors), [
        (span.name, span.status.description) for span in errors
    ]
    events = [event for span in spans for event in span.events if event.name == "exception"]
    assert events
    assert all(
        dict(event.attributes or {}) == {"exception.type": "RuntimeError"} for event in events
    )
    serialized = json.dumps(
        [
            {
                "status": span.status.description,
                "attributes": dict(span.attributes or {}),
                "events": [
                    {"name": event.name, "attributes": dict(event.attributes or {})}
                    for event in span.events
                ],
            }
            for span in spans
        ]
    )
    assert secret not in serialized
    assert received
    assert all(secret not in snapshot for snapshot in received)
    assert all(secret not in event.model_dump_json() for event in all_events(cfg))


@pytest.mark.parametrize(
    ("status_error", "event_name", "removed", "expected_type"),
    [
        (False, "plain", {}, None),
        (False, "plain", {"app.error_budget": 3}, None),
        (False, "plain", {"error.detail": "SECRET"}, None),
        (False, "plain", {"exception.message": "SECRET"}, "Exception"),
        (False, "plain", {"error.type": "ValueError"}, "ValueError"),
        (True, "plain", {}, "Exception"),
        (False, "exception", {}, None),
    ],
)
def test_error_boundary_preserves_non_error_attributes(
    status_error: bool, event_name: str, removed: dict[str, Any], expected_type: str | None
) -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import Status, StatusCode

    exporter = InMemorySpanExporter()
    boundary = _langfuse._error_boundary()
    boundary.add_span_processor(SimpleSpanProcessor(exporter))
    provider = TracerProvider(active_span_processor=boundary, shutdown_on_exit=False)
    safe = {"keep": "ok", "count": 3, "enabled": True}
    try:
        with provider.get_tracer("regression").start_as_current_span(
            "ordinary", attributes=safe | removed
        ) as span:
            span.add_event(event_name, safe | removed)
            if status_error:
                span.set_status(Status(StatusCode.ERROR, "SECRET"))
        (result,) = exporter.get_finished_spans()
        expected = safe | ({"exception.type": expected_type} if expected_type else {})
        assert dict(result.attributes or {}) == expected
        event_expected = (
            {"exception.type": "Exception"}
            if event_name == "exception"
            else safe | ({"exception.type": expected_type} if removed and expected_type else {})
        )
        assert dict(result.events[0].attributes or {}) == event_expected
        assert result.status.status_code is (StatusCode.ERROR if status_error else StatusCode.UNSET)
        assert result.status.description == ("Exception" if status_error else None)
    finally:
        provider.shutdown()


def test_error_boundary_preserves_dropped_counts() -> None:
    from opentelemetry.sdk.trace import Span, SpanLimits, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import Link, SpanContext, TraceFlags

    exporter = InMemorySpanExporter()
    boundary = _langfuse._error_boundary()
    boundary.add_span_processor(SimpleSpanProcessor(exporter))
    provider = TracerProvider(
        active_span_processor=boundary,
        shutdown_on_exit=False,
        span_limits=SpanLimits(max_span_attributes=2, max_events=2, max_links=1),
    )
    parent = SpanContext(1, 2, False, TraceFlags(1))
    try:
        span = provider.get_tracer("regression", "1").start_span(
            "limited", links=[Link(parent), Link(parent)]
        )
        assert isinstance(span, Span)
        for i in range(4):
            span.set_attribute(f"keep.{i}", i)
            span.add_event(f"event.{i}")
        expected = (span.dropped_attributes, span.dropped_events, span.dropped_links)
        assert expected == (2, 2, 1)
        span.end()
        (result,) = exporter.get_finished_spans()
        assert (result.dropped_attributes, result.dropped_events, result.dropped_links) == expected
        assert dict(result.attributes or {}) == dict(span.attributes or {})
        assert [event.name for event in result.events] == [event.name for event in span.events]
        assert result.links == span.links
        assert result.context == span.context
        assert result.instrumentation_scope == span.instrumentation_scope
    finally:
        provider.shutdown()


@pytest.mark.parametrize("kind", ["function", "custom"])
async def test_tool_and_custom_span_errors_are_value_free(
    tmp_path: Path, exported: Any, kind: str
) -> None:
    import json

    from opentelemetry.trace import StatusCode

    secret = "TOOL_CUSTOM_ERROR_SECRET_b26f"
    async with open_llm_run(
        tmp_path, config=manager_config(tracing=TracingConfig(langfuse=True))
    ) as run:
        scope = run.scope("manager")
        with (
            llm_scope(scope),
            _langfuse.component_span(scope),
            agents.trace("component errors"),
        ):
            child = (
                agents.function_span("tool failure")
                if kind == "function"
                else agents.custom_span("component failure")
            )
            with child:
                child.set_error({"message": secret, "data": {"detail": secret}})
    exported[1][0].flush()
    spans = exported[0].get_finished_spans()
    errors = [span for span in spans if span.status.status_code is StatusCode.ERROR]
    assert len(errors) == 1
    assert errors[0].name == ("tool failure" if kind == "function" else "component failure")
    assert errors[0].status.description == "Exception"
    assert errors[0].attributes["exception.type"] == "Exception"
    assert secret not in json.dumps([span.to_json() for span in spans])
