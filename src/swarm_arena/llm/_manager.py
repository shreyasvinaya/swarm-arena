"""Run stateless, metered manager conversations through the Agents SDK."""

from typing import Any, cast

import anyio
from agents import Agent, ModelSettings, Runner
from agents import RunConfig as SDKRunConfig
from agents.agent import StopAtTools
from agents.exceptions import MaxTurnsExceeded
from pydantic import Field

from swarm_arena._agent.context import AgentContext, _local_timer_scope
from swarm_arena._agent.manager import LimitReached, Manager
from swarm_arena._budget.llm_scope import LLMScope, current_llm_scope
from swarm_arena._errors import _core_error, _is_core_control, _stop_cleanup_error
from swarm_arena._journal.llm_record import drain
from swarm_arena._model.data import FiniteJsonObject
from swarm_arena._model.planning_types import RoundReport
from swarm_arena._model.ref import _ConfigInputModel, _validate_config, config_or_fields
from swarm_arena._outcomes import _Outcome
from swarm_arena._store.io import _storage_io
from swarm_arena.llm._actions import _own_actions, build_tools
from swarm_arena.llm._cache import cache_namespace
from swarm_arena.llm._catalog import build_model
from swarm_arena.llm._hooks import BudgetExceeded, BudgetHooks, MeteredModel, _prepare, _sdk_control
from swarm_arena.llm._linking import CallLinks, ManagerDeps
from swarm_arena.llm._prompts import manager_instructions, reference_files, round_input


async def _interrupted_drain(scope: LLMScope) -> None:
    """Skip report waiting and bound shielded accounting after interruption."""
    try:
        with anyio.fail_after(scope.settle_timeout_s, shield=True):
            await drain(scope, wait=False)
    except TimeoutError:
        raise _stop_cleanup_error() from None


async def _finish_accounting(scope: LLMScope, *, interrupted: bool) -> None:
    if interrupted:
        await _interrupted_drain(scope)
        return
    try:
        await drain(scope, wait=True)
    except anyio.get_cancelled_exc_class():
        # Cancellation may arrive while waiting for a provider callback, after
        # Runner has returned. That exit still owns conservative settlement.
        await _interrupted_drain(scope)
        raise


class AgentsSDKManager(Manager):
    """
    Choose actions with a fresh, budgeted Agents SDK conversation each round.

    The core supplies an LLM scope and authorized AgentContext. Conversation
    history is discarded between rounds; private notes and fixed board views
    carry useful state forward. Calling ``finish_round`` ends the conversation.
    Every exit drains call accounting before returning to the scope owner.

    Parameters
    ----------
    config : AgentsSDKManager.Config or None, optional
        Explicit validated settings. Omit when supplying keyword fields.
    **fields : Any
        Config fields, including the required model catalog alias.

    Examples
    --------
    >>> from swarm_arena import Ref
    >>> from swarm_arena.manager import Manager
    >>> manager = Ref(path="swarm_arena.llm:AgentsSDKManager",
    ...     config={"model": "opus"}).build(Manager)
    >>> manager.models()
    ('opus',)
    """

    class Config(_ConfigInputModel):
        """
        Configure the model alias, stable instructions and action behavior.

        Parameters
        ----------
        model : str
            Human-approved model catalog alias with the manager role.
        instructions : str, default=''
            Stable instructions placed before changing round facts.
        seed_inputs : dict[str, Any], optional
            Finite JSON starting inputs, defaulting to an empty object.
        temperature : float or None, optional
            Optional finite sampling temperature.
        action_timeout_s : float, default=60.0
            Positive timeout for actions other than core-timed tool runs.
        parallel_tool_calls : bool, default=True
            Allow the SDK model to request concurrent function calls.

        Examples
        --------
        >>> from swarm_arena.llm import AgentsSDKManager
        >>> AgentsSDKManager.Config(model="opus").action_timeout_s
        60.0
        """

        model: str
        instructions: str = ""
        seed_inputs: FiniteJsonObject = {}
        temperature: float | None = None
        action_timeout_s: float = Field(default=60.0, gt=0)
        parallel_tool_calls: bool = True

    def __init__(self, config: Config | None = None, /, **fields: Any) -> None:
        """Validate explicit configuration or keyword fields at the safe boundary."""
        self._config = _validate_config(
            self.Config, config_or_fields(config, fields), where="AgentsSDKManager"
        )

    @property
    def config(self) -> Config:
        """
        Return the validated manager settings.

        Returns
        -------
        AgentsSDKManager.Config
            Model, prompt and action settings.

        Examples
        --------
        >>> from swarm_arena.llm import AgentsSDKManager
        >>> AgentsSDKManager(model="opus").config.parallel_tool_calls
        True
        """
        return cast(AgentsSDKManager.Config, self._config)

    def models(self) -> tuple[str, ...]:
        """
        Return the model alias used for admission and fingerprint checks.

        Returns
        -------
        tuple[str, ...]
            The configured model alias as a one-element tuple.

        Examples
        --------
        >>> from swarm_arena.llm import AgentsSDKManager
        >>> AgentsSDKManager(model="opus").models()
        ('opus',)
        """
        return (self.config.model,)

    async def run_round(self, ctx: AgentContext) -> RoundReport:
        """
        Run a fresh conversation and settle every admitted call on exit.

        Parameters
        ----------
        ctx : AgentContext
            Authorized actions and the round's fixed, redacted board views.

        Returns
        -------
        RoundReport
            The finish_round summary, or the model's final text.

        Raises
        ------
        LimitReached
            If the round's turn, time or core action limit is reached.
        BudgetExceeded
            If the model's reservation is refused for budget.
        RuntimeError
            If no manager scope is active or owned cleanup fails.
        ValueError
            If the catalog alias lacks the manager role.

        Examples
        --------
        The engine supplies ``ctx`` and the owned accounting scope.

        >>> from swarm_arena.llm import AgentsSDKManager
        >>> report = await AgentsSDKManager(model="opus").run_round(ctx)  # doctest: +SKIP
        """
        scope = current_llm_scope()
        if scope is None or scope.role != "manager":
            raise _core_error(RuntimeError, "manager requires an active manager LLM scope")
        outcome = _Outcome()
        failure: BaseException | None = None
        interrupted = False
        report: RoundReport | None = None
        timer: anyio.CancelScope | None = None
        try:
            spec = await _prepare(lambda: scope.models.resolve(self.config.model, "manager"))
            reference = await _storage_io(
                reference_files, ctx.goal, scope.artifacts, site="manager reference files"
            )
            deps = ManagerDeps(ctx=ctx, links=CallLinks())

            def prepare() -> tuple[Agent[ManagerDeps], str]:
                agent = Agent[ManagerDeps](
                    name="swarm-arena manager",
                    instructions=manager_instructions(
                        instructions=self.config.instructions,
                        tool_description=ctx.tool_description,
                        goal=ctx.goal,
                        reference=reference,
                    ),
                    tools=[*build_tools(ctx, deps, action_timeout_s=self.config.action_timeout_s)],
                    model=MeteredModel(
                        build_model(spec),
                        alias=self.config.model,
                        spec=spec,
                        namespace=cache_namespace(scope.cache, ctx.goal.namespace),
                        links=deps.links,
                        temperature=self.config.temperature,
                    ),
                    model_settings=ModelSettings(
                        parallel_tool_calls=self.config.parallel_tool_calls
                    ),
                    tool_use_behavior=StopAtTools(stop_at_tool_names=["finish_round"]),
                )
                return agent, round_input(ctx, seed_inputs=self.config.seed_inputs)

            agent, input_text = await _prepare(prepare)
            timer = anyio.CancelScope(
                deadline=anyio.current_time() + scope.limits.effective_round_timeout_s()
            )
            with _local_timer_scope(timer, relay_provider_stop=True) as local_timer:
                async with anyio.create_task_group() as group:
                    with _own_actions(group, local_timer) as actions:
                        try:
                            with timer:
                                result = await Runner.run(
                                    agent,
                                    input_text,
                                    context=deps,
                                    max_turns=scope.limits.max_turns_per_round,
                                    hooks=BudgetHooks(),
                                    run_config=SDKRunConfig(
                                        tracing_disabled=not scope.tracing.langfuse
                                    ),
                                )
                        except BaseException as error:
                            failure = error
                        actions.accepting = False
                        if failure is not None or timer.cancel_called:
                            actions.stop(caller=False)
                        try:
                            outcome.propagate_control(RuntimeError())
                        except anyio.get_cancelled_exc_class() as error:
                            failure = error
                            actions.stop(caller=True)
                            group.cancel_scope.shield = True
                        while True:
                            try:
                                await actions.wait()
                                break
                            except anyio.get_cancelled_exc_class() as error:
                                failure = error
                                actions.stop(caller=True)
                                group.cancel_scope.shield = True
                        group.cancel_scope.shield = True
                if local_timer.failure is not None:
                    failure = (
                        BaseExceptionGroup("manager action unwind", [failure, local_timer.failure])
                        if failure is not None
                        else local_timer.failure
                    )
                if failure is not None:
                    raise failure
            if timer.cancel_called:
                raise TimeoutError
            report = RoundReport(
                summary=deps.summary if deps.summary is not None else str(result.final_output or "")
            )
        except BaseException as error:
            failure = error
            interrupted = (timer is not None and timer.cancel_called) or isinstance(
                error, (TimeoutError, anyio.get_cancelled_exc_class())
            )
        try:
            await _finish_accounting(scope, interrupted=interrupted)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as cleanup_error:
            failure = (
                BaseExceptionGroup("manager round cleanup", [failure, cleanup_error])
                if failure is not None
                else cleanup_error
            )
        if failure is not None:
            classified = outcome.classify(failure)
            if _is_core_control(classified):
                # Keep the documented public budget carrier for direct callers;
                # the shared engine classifier unwraps its saved core refusal.
                if type(failure) is BudgetExceeded:
                    carrier = _sdk_control(classified)
                    assert carrier is not None
                    raise carrier from None
                raise classified from None
            if isinstance(classified, MaxTurnsExceeded):
                raise _core_error(
                    LimitReached, "max_turns_per_round", "manager turn limit reached"
                ) from None
            if isinstance(classified, TimeoutError) or (timer is not None and timer.cancel_called):
                raise _core_error(
                    LimitReached, "round_timeout_s", "manager round timed out"
                ) from None
            outcome.contain(classified, RuntimeError, "manager round")
        # Also observe a caller cancellation swallowed by an SDK callback.
        outcome.propagate_control(RuntimeError())
        assert report is not None
        return report
