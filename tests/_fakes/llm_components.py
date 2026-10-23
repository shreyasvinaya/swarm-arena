"""Import-triggered LLM integration fixtures for admission and startup tests."""

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import anyio
import pytest
from agents import AgentOutputSchemaBase, Handoff, ModelResponse, ModelSettings, Tool
from agents.items import TResponseInputItem
from agents.models.interface import ModelTracing
from openai.types.responses.response_prompt_param import ResponsePromptParam

import swarm_arena.llm  # noqa: F401 - register optional model and price hooks
from swarm_arena._budget.llm_scope import LLMScope
from swarm_arena._journal.llm_record import pending_scope, record_call
from swarm_arena._model.base import _SafeModel
from swarm_arena._model.goal import Goal
from swarm_arena._model.results import Candidate, Score
from swarm_arena.board import BoardView, Curator, Learning
from swarm_arena.llm import AgentsSDKManager
from swarm_arena.llm._catalog import call_cost
from swarm_arena.llm._linking import tool_calls_of
from swarm_arena.manager import AgentContext, Manager, RoundReport
from swarm_arena.scoring import Judge, JudgeContext
from swarm_arena.testing import FakeModel, FakeTurn
from swarm_arena.tool import Usage


class AliasManager(Manager):
    """Declare one catalog model without making provider calls."""

    class Config(_SafeModel):
        model: str

    def models(self) -> tuple[str, ...]:
        """Return the configured alias."""
        return (self.Config.model_validate(self.config).model,)

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        """Complete without work."""
        return RoundReport()


class AliasJudge(Judge):
    """Declare one catalog model without making provider calls."""

    class Config(_SafeModel):
        model: str

    def models(self) -> tuple[str, ...]:
        """Return the configured alias."""
        return (self.Config.model_validate(self.config).model,)

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        """Return no scores."""
        return []


class AliasCurator(Curator):
    """Declare one startup model without making provider calls."""

    class Config(_SafeModel):
        model: str

    def models(self) -> tuple[str, ...]:
        """Return the configured alias."""
        return (self.Config.model_validate(self.config).model,)

    async def curate(
        self, namespace: str, new: Sequence[Learning], view: BoardView
    ) -> list[Learning]:
        """Return no summaries."""
        return []


class FakeSDKManager(AgentsSDKManager):
    """Register a per-construction offline conversation, including on resume."""

    def __init__(self, config: AgentsSDKManager.Config | None = None, /, **fields: Any) -> None:
        super().__init__(config, **fields)
        self.fake = FakeModel(
            [
                FakeTurn(tool_calls=(("my_candidates", {}),)),
                FakeTurn(tool_calls=(("finish_round", {"summary": "offline round"}),)),
            ]
        )
        self.fake.register(self.config.model)


class LateFakeModel(FakeModel):
    """Return immediately and record usage later, or never when delay_s is None.

    Tests may supply ``release`` to hold each callback until a controlled event.
    ``ready`` signals that a response has been returned for a metered call;
    ``recorded`` signals callback completion. Owners must await ``flush`` or
    ``aclose`` before releasing their run services. No task is started when
    delay_s is None.
    """

    def __init__(
        self,
        turns: Sequence[FakeTurn],
        *,
        delay_s: float | None,
        provider: str = "fake",
        release: anyio.Event | None = None,
    ) -> None:
        super().__init__(turns, provider=provider)
        self.delay_s = delay_s
        self.release = release
        self.ready = anyio.Event()
        self.recorded = anyio.Event()
        self.callbacks: list[asyncio.Task[None]] = []

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
        metadata = dict(model_settings.metadata or {})
        # Preserve FakeModel's isolated scripts and response validation while
        # suppressing only its synchronous recorder invocation.
        response = await super().get_response(
            system_instructions,
            input,
            replace(model_settings, metadata=None),
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )
        self.calls[-1]["model_settings"] = model_settings
        scope = pending_scope(metadata)
        if scope is not None and self.delay_s is not None:
            details = response.usage.input_tokens_details
            usage = Usage(
                input_tokens=response.usage.input_tokens
                - details.cached_tokens
                - details.cache_write_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=details.cached_tokens,
                cache_write_tokens=details.cache_write_tokens,
            )
            spec = scope.models.root[metadata["model_alias"]]
            kwargs = dict(
                metadata=metadata,
                model=spec.model,
                provider=self.provider,
                usage=usage,
                cost_usd=call_cost(spec, usage, litellm_cost=None),
                cache_hit=False,
                latency_ms=0,
                status="ok",
                error=None,
                request={"system": system_instructions, "input": input},
                response={"output": [item.model_dump(mode="json") for item in response.output]},
                litellm_call_id=None,
                tool_call_ids=tool_calls_of(response),
                settle=True,
            )
            self.callbacks.append(asyncio.create_task(self._record(kwargs)))
        self.ready.set()
        return response

    async def _record(self, kwargs: dict[str, Any]) -> None:
        if self.release is not None:
            await self.release.wait()
        assert self.delay_s is not None
        await anyio.sleep(self.delay_s)
        await record_call(**kwargs)
        self.recorded.set()

    async def flush(self) -> None:
        """Join every scheduled callback and propagate callback failures."""
        if self.callbacks:
            await asyncio.gather(*self.callbacks)
        self.callbacks.clear()

    async def aclose(self) -> None:
        """Cancel and join outstanding callbacks without leaking tasks."""
        for task in self.callbacks:
            task.cancel()
        if self.callbacks:
            await asyncio.gather(*self.callbacks, return_exceptions=True)
        self.callbacks.clear()


def missing_callback_timeout(patch: pytest.MonkeyPatch) -> None:
    """
    Give fabricated provider failures a short report wait, preserving cleanup.

    Parameters
    ----------
    patch : pytest.MonkeyPatch
        Test-owned patch lifetime. Use only for fakes that never emit their
        usage callback; normal run/journal cleanup keeps RunConfig's allowance.

    Examples
    --------
    >>> with pytest.MonkeyPatch.context() as patch:
    ...     missing_callback_timeout(patch)
    """
    from swarm_arena._engine import round as round_module

    original_scope = round_module.LLMScope
    original_child = LLMScope.child

    def child(parent: LLMScope) -> LLMScope:
        return replace(original_child(parent), settle_timeout_s=2)

    def scope(**fields: Any) -> LLMScope:
        fields["settle_timeout_s"] = 2
        return original_scope(**fields)

    patch.setattr(LLMScope, "child", child)
    patch.setattr(round_module, "LLMScope", scope)
